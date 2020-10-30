import itertools

from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response

from clustering.kmeans import KMeans
from ir_engine import SearchEngine
import index
import search_engine
import time

try:
    engine = SearchEngine()
except:
    print("No index found")


@api_view(['POST'])
def process_query(request: Request):
    inp = time.time()
    global engine
    start_time = time.time()
    result = search_engine.do_search(engine, request.data, start_time)
    outp = time.time()
    print(inp, outp, outp - inp)
    return Response(result)


@api_view(['POST'])
def initialize_indices(request: Request):
    global engine
    index.indexing()
    engine = SearchEngine()
    return Response()


@api_view(['GET'])
def get_doc(request: Request):
    doc_id = request.query_params['id']
    return Response(index.fetch_result([int(doc_id)]))


@api_view(['GET'])
def get_most_similar_docs(request: Request, doc_id):
    kmeans = KMeans.get_k_means()
    label = kmeans.labels_list[kmeans.k_index][doc_id]
    all_docs_in_cluster = kmeans.get_cluster_data(label)
    doc_vec = engine.dictionary.doc2vec[doc_id]
    resses = list(itertools.zip_longest(all_docs_in_cluster, [], fillvalue=[]))
    most_similars = engine.k_most_similarity(4, resses, None, doc_vec)[1:]
    return Response(index.fetch_result(list(zip(*most_similars))[0]))


@api_view(['POST'])
def to_db(request):
    index.to_db()
    return Response()
