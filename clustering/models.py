from django.db import models
from django.contrib.auth.models import User


class AnalysisSession(models.Model):
    """분석 세션: 사용자가 업로드한 CSV 파일 1건에 대한 분석 결과"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    created_at = models.DateTimeField(auto_now_add=True)
    file_name = models.CharField(max_length=255, blank=True)
    total_customers = models.IntegerField(default=0)
    n_clusters = models.IntegerField(default=3)

    class Meta:
        ordering = ['-created_at']
        verbose_name = '분석 세션'
        verbose_name_plural = '분석 세션 목록'

    def __str__(self):
        return f"[{self.user.username}] {self.file_name} ({self.created_at:%Y-%m-%d %H:%M})"


class CustomerCluster(models.Model):
    """군집별 집계 결과 및 추천 정보"""
    CLUSTER_TYPES = [
        ('vip', 'VIP 고객군'),
        ('churn_risk', '이탈 위험군'),
        ('potential', '잠재 고객군'),
        ('general', '일반 고객군'),
    ]

    session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name='clusters')
    cluster_label = models.CharField(max_length=20, choices=CLUSTER_TYPES)
    cluster_index = models.IntegerField()
    customer_count = models.IntegerField(default=0)

    avg_recency = models.FloatField(default=0)
    avg_frequency = models.FloatField(default=0)
    avg_monetary = models.FloatField(default=0)

    recommendation_channel = models.TextField(blank=True)
    recommendation_keywords = models.TextField(blank=True)
    recommendation_message = models.TextField(blank=True)

    # 산점도용 JSON 데이터 (각 고객의 R/F/M 값)
    scatter_data_json = models.TextField(blank=True, default='[]')

    class Meta:
        ordering = ['cluster_index']
        verbose_name = '고객 군집'
        verbose_name_plural = '고객 군집 목록'

    def __str__(self):
        return f"{self.session} - {self.get_cluster_label_display()} ({self.customer_count}명)"
