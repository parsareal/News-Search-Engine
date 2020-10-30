import time
import requests
import feedparser
import heapq
from dictionary import Dictionary
from pre_processor import DocTokenizer
import random
import pandas as pd
import threading

dataset = pd.DataFrame(columns=["id", "title", "summary", "publish_date", "content", "url", "thumbnail"])
RSS_PATH = "./Rss.csv"
NUM_BACKQ = 3
NUM_FRONTQ = 0
POLITENESS_RATE = 20
FRESHNESS_BASIC_INTERVAL = 60 * 10
tokenizer = DocTokenizer()
FRONT_QUEUES = []
CRAWLED_ENTRY = set()
NEWS_HTML = []
RSS2Q = dict()
docId = 0
BACK_HEAP = []
FRONT_QUEUES_FRESHNESS_INTERVAL = {}
IDLE_BACK_QUEUES = {i for i in range(NUM_BACKQ)}
BACK_QUEUES = [list() for i in range(NUM_BACKQ)]
dictionary = Dictionary()
isFinish = False


def prioritize_rss():
    global NUM_FRONTQ, FRONT_QUEUES, FRONT_QUEUES_FRESHNESS_INTERVAL
    rss = []
    num_frontQ = 0
    rss_file = open(RSS_PATH, 'r').readlines()
    for line in rss_file:
        rss_item = line.split(',')
        print(rss_item)
        priority = int(rss_item[1].strip('\n'))
        if priority > num_frontQ:
            num_frontQ = priority
        rss.append((rss_item[0], priority))
    NUM_FRONTQ = num_frontQ + 1
    FRONT_QUEUES = [list() for i in range(NUM_FRONTQ)]
    for t in rss:
        FRONT_QUEUES_FRESHNESS_INTERVAL[t[0]] = FRESHNESS_BASIC_INTERVAL
        next_time = time.time()
        FRONT_QUEUES[t[1]].append((t[0], next_time))


def remove_duplicate_url(rss_url):
    newsFeed = feedparser.parse(rss_url)
    unseen_entry = []
    for entry in newsFeed.entries:
        if entry not in CRAWLED_ENTRY:
            CRAWLED_ENTRY.add(entry)
            unseen_entry.append(entry)
    if len(unseen_entry) > 1:
        FRONT_QUEUES_FRESHNESS_INTERVAL[rss_url] -= 60
    elif len(unseen_entry) == 0:
        FRONT_QUEUES_FRESHNESS_INTERVAL[rss_url] += 60
    print("------------")
    print(rss_url)
    print(len(unseen_entry))
    print("------------")
    return unseen_entry


def select_forntQ():
    global FRONT_QUEUES
    for i in range(NUM_FRONTQ)[::-1]:
        if len(FRONT_QUEUES[i]) > 0:
            for j in range(len(FRONT_QUEUES[i])):
                if (FRONT_QUEUES[i][j])[1] < time.time():
                    FRONT_QUEUES[i][j] = (
                        FRONT_QUEUES[i][j][0], time.time() + FRONT_QUEUES_FRESHNESS_INTERVAL[FRONT_QUEUES[i][j][0]])
                    return FRONT_QUEUES[i][j][0]
    return None


def index_doc(doc, id):
    print("Doc{} start processing ...".format(id), end="\t")
    positionals = tokenizer.get_tokens(doc)
    terms = []
    for positional in positionals:
        terms.append((positional[0], len(positional[1])))
        if dictionary.existed_in_dictionary(positional[0]):
            dictionary.add_term_to_dictionary(positional, id)
            temp = dictionary.terms_cf[positional[0]]
        else:
            new_posting = list()
            new_posting.append((id, positional[1]))
            dictionary.dictionary[positional[0]] = (1, new_posting)
            temp = 0
        dictionary.terms_cf[positional[0]] = len(positional[1]) + temp
    print("Done")


def init_backQ():
    global dataset, docId
    while len(IDLE_BACK_QUEUES) > 0:
        new_rss = select_forntQ()
        entries = remove_duplicate_url(new_rss)
        queue_index = None
        if new_rss not in RSS2Q.keys():
            queue_index = IDLE_BACK_QUEUES.pop()
            RSS2Q[new_rss] = queue_index
            heapq.heappush(BACK_HEAP, (time.time() + POLITENESS_RATE, queue_index))
        else:
            queue_index = RSS2Q[new_rss]
        for entry in entries:
            dataset = dataset.append({
                'title': entry.title,
                'summary': entry.summary,
                'publish_data': entry.published,
                'url': entry.link,
                'content': '',
                'id': int(docId)
            }, ignore_index=True)
            BACK_QUEUES[queue_index].append((docId, entry.link))
            docId += 1


def update_backQ():
    global RSS2Q, docId, dataset
    if len(BACK_HEAP) > 0:
        item = heapq.heappop(BACK_HEAP)
        id, f_link = BACK_QUEUES[item[1]].pop()
        # print(f_link)
        res = requests.get(f_link)
        NEWS_HTML.append((id, res.text))
        if len(BACK_QUEUES[item[1]]) > 0:
            heapq.heappush(BACK_HEAP, (time.time() + POLITENESS_RATE + random.randint(1, 5), item[1]))
            return False
        else:
            IDLE_BACK_QUEUES.add(item[1])
            RSS2Q = {key: val for key, val in RSS2Q.items() if val != item[1]}
            new_rss = select_forntQ()
            if new_rss is None:
                print("All hosts are waiting ...")
                return False
            entries = remove_duplicate_url(new_rss)
            if len(entries) > 0:
                if new_rss not in RSS2Q.keys():
                    queue_index = IDLE_BACK_QUEUES.pop()
                    RSS2Q[new_rss] = queue_index
                    heapq.heappush(BACK_HEAP, (time.time() + POLITENESS_RATE + random.randint(5, 250), queue_index))
                else:
                    queue_index = RSS2Q[new_rss]
                for entry in entries:
                    dataset = dataset.append({
                        'title': entry.title,
                        'summary': entry.summary,
                        'publish_data': entry.published,
                        'url': entry.link,
                        'content': '',
                        'id': int(docId)
                    }, ignore_index=True)
                    docId += 1
                    BACK_QUEUES[queue_index].append((docId, entry.link))
            return False
    else:
        return True


def crawling():
    global isFinish
    prioritize_rss()
    init_backQ()
    while not isFinish:
        isFinish = update_backQ()


def onlineIndexer():
    global dataset, isFinish, dictionary
    while len(NEWS_HTML) > 0 or not isFinish:
        if len(NEWS_HTML) > 0:
            news = NEWS_HTML.pop()
            dataset.loc[dataset["id"] == news[0], 'content'] = news[1]
            index_doc(news[1], news[0])
    dict = dictionary.get_dictionary()
    with open('../index_crawler.txt', 'w', encoding="utf_8") as f:
        for key in sorted(dict.keys()):
            f.writelines([key, " => ", str(dict[key]), "\n"])


crawler = threading.Thread(target=crawling)
indexer = threading.Thread(target=onlineIndexer)
crawler.start()
indexer.start()

# def test():
#     rss_file = open(RSS_PATH, 'r').readlines()
#     for line in rss_file:
#         news = feedparser.parse(line.split(',')[0])
#         if len(news.entries) > 0:
#             print(news.entries[0].keys())
#             print("----------------------------------------------------")
#
#
# test()
