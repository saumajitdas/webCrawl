import re
import nltk
from html.parser import HTMLParser
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

# Download required data
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

lemmatizer = WordNetLemmatizer()

class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for (key, value) in attrs:
                if key == 'href' and value.startswith('http'):
                    self.links.append(value)

def is_named_entity(phrase):
    try:
        tokens = word_tokenize(phrase)
        tags = pos_tag(tokens)
        chunk = ne_chunk(tags)
        return any(isinstance(subtree, Tree) for subtree in chunk)
    except:
        return False

def is_meaningful(phrase):
    phrase_norm = ' '.join([lemmatizer.lemmatize(w.lower(), pos='n') for w in phrase.split()])
    return bool(wn.synsets(phrase_norm.replace(' ', '_'))) or bool(wn.synsets(phrase_norm)) or is_named_entity(phrase)

class CrawlNode:
    def __init__(self, url, depth=0):
        self.url = url
        self.left = None
        self.right = None
        self.depth = depth
        self.mono = []
        self.bi = []
        self.tri = []

MAX_PAGES = 10
pages_crawled = 0

def crawl(node, max_depth=1, visited=None):
    global pages_crawled
    if visited is None:
        visited = set()

    if node.depth > max_depth or node.url in visited or pages_crawled >= MAX_PAGES:
        return

    visited.add(node.url)
    pages_crawled += 1

    print(f"\n[Crawling] {node.url}")

    try:
        req = Request(node.url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urlopen(req, timeout=5)
        html = response.read().decode('utf-8', errors='ignore')
    except (URLError, HTTPError) as e:
        print(f"  [Error] Could not fetch: {e}")
        return

    html = re.sub(r'<(script|style|noscript).*?>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<head.*?>.*?</head>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html)

    words = re.findall(r'\b\w+\b', text)
    print(f"  [Info] Total word count: {len(words)}")

    # First pass: mono
    for word in words:
        if word.isalpha():
            lemma = lemmatizer.lemmatize(word.lower(), pos='n')
            if is_meaningful(lemma) and lemma not in node.mono:
                node.mono.append(lemma)

    # Second pass: bi and tri + mono fallback
    for i in range(len(words)):
        if i + 1 < len(words):
            w1, w2 = words[i], words[i + 1]
            bi = f"{w1} {w2}"
            if is_meaningful(bi) and bi not in node.bi:
                node.bi.append(bi)
            for word in [w1, w2]:
                if word.isalpha():
                    lemma = lemmatizer.lemmatize(word.lower(), pos='n')
                    if is_meaningful(lemma) and lemma not in node.mono:
                        node.mono.append(lemma)

        if i + 2 < len(words):
            w1, w2, w3 = words[i], words[i + 1], words[i + 2]
            tri = f"{w1} {w2} {w3}"
            if is_meaningful(tri) and tri not in node.tri:
                node.tri.append(tri)
            for word in [w1, w2, w3]:
                if word.isalpha():
                    lemma = lemmatizer.lemmatize(word.lower(), pos='n')
                    if is_meaningful(lemma) and lemma not in node.mono:
                        node.mono.append(lemma)

    print(f"  [Info] Found unique words - Mono: {len(node.mono)}, Bi: {len(node.bi)}, Tri: {len(node.tri)}")

    parser = LinkExtractor()
    try:
        parser.feed(html)
    except:
        print("  [Warning] HTML parsing failed.")
        return

    links = parser.links[:2]
    if len(links) > 0:
        node.left = CrawlNode(links[0], node.depth + 1)
        crawl(node.left, max_depth, visited)
    if len(links) > 1:
        node.right = CrawlNode(links[1], node.depth + 1)
        crawl(node.right, max_depth, visited)

def print_tree_pretty(node, level=0, branch="Root"):
    if node is None:
        return
    if not (node.mono or node.bi or node.tri):
        print("  [Note] Skipping empty node:", node.url)
        return
    indent = "  " * level
    print(f"{indent}{branch} - {node.url}")
    print(f"{indent}  Top 5 Mono: {', '.join(node.mono[:5])}")
    print(f"{indent}  Top 3 Bi: {', '.join(node.bi[:3])}")
    print(f"{indent}  Top 2 Tri: {', '.join(node.tri[:2])}")
    print_tree_pretty(node.left, level + 1, "L")
    print_tree_pretty(node.right, level + 1, "R")

# --- Entry point ---
if __name__ == '__main__':
    start_url = "https://www.bbc.com/news"
    root = CrawlNode(start_url, depth=0)
    crawl(root, max_depth=10)
    print("\n\n[Final Tree Output]")
    print_tree_pretty(root)
