from django.apps import AppConfig


class ClusteringConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'clustering'
    verbose_name = '고객 군집화'
