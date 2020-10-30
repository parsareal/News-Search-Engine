from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from indexer import views

urlpatterns = [
    path('process', views.process_query),
    path('setup', views.initialize_indices),
    path('doc', views.get_doc),
    path('db', views.to_db),
    path('<int:doc_id>', views.get_most_similar_docs),
]

urlpatterns = format_suffix_patterns(urlpatterns)
