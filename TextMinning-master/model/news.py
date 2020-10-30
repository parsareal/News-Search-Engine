from mongoengine import Document, StringField, IntField, ListField


class NewsModel(Document):
    meta = {'allow_inheritance': True}
    news_id = IntField(default=0, null=False, unique=True)
    content = StringField(default='', null=False)
    title = StringField(default='', null=False)
    thumbnail = StringField(default='', null=False)
    meta_tags = ListField()
    publish_date = StringField(default='', null=False)
    url = StringField(default='', null=False)
    summary = StringField(default='', null=False)