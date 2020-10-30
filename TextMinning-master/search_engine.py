import query_processor
import ir_engine

processor = query_processor.QueryProcessor()

def do_search(search_engine, user_query, start_time):
    global processor
    cat = None
    source = None
    if 'cat' in user_query:
        cat = user_query["cat"]
    if 'source' in user_query:
        source = user_query["source"]
    jquery = processor.parse(user_query['query'], cat, source)
    result = search_engine.search(jquery, start_time)
    return result
