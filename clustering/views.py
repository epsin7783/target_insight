import json
import csv
import io
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.views.decorators.http import require_POST

from .services import run_rfm_clustering, generate_sample_csv
from .models import AnalysisSession, CustomerCluster


def index(request):
    return render(request, 'clustering/home.html')


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'환영합니다, {user.username}님!')
            return redirect('dashboard')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = UserCreationForm()
    return render(request, 'clustering/signup.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, '아이디 또는 비밀번호가 올바르지 않습니다.')
    else:
        form = AuthenticationForm()
    return render(request, 'clustering/login.html', {'form': form})


@login_required
def dashboard_view(request):
    sessions = AnalysisSession.objects.filter(user=request.user).prefetch_related('clusters')[:10]
    return render(request, 'clustering/dashboard.html', {'sessions': sessions})


@login_required
@require_POST
def analyze_view(request):
    csv_file = request.FILES.get('csv_file')
    n_clusters = int(request.POST.get('n_clusters', 3))

    if not csv_file:
        messages.error(request, 'CSV 파일을 업로드해 주세요.')
        return redirect('dashboard')

    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'CSV 파일만 업로드 가능합니다.')
        return redirect('dashboard')

    if n_clusters not in [3, 4]:
        n_clusters = 3

    try:
        result = run_rfm_clustering(csv_file, n_clusters)
    except ValueError as e:
        messages.error(request, str(e))
        return redirect('dashboard')
    except Exception as e:
        messages.error(request, f'분석 중 오류가 발생했습니다: {e}')
        return redirect('dashboard')

    # DB 저장
    session = AnalysisSession.objects.create(
        user=request.user,
        file_name=csv_file.name,
        total_customers=result['total'],
        n_clusters=result['n_clusters'],
    )

    cluster_objs = []
    for c in result['clusters']:
        cluster_objs.append(CustomerCluster(
            session=session,
            cluster_label=c['label'],
            cluster_index=c['cluster_index'],
            customer_count=c['count'],
            avg_recency=c['avg_r'],
            avg_frequency=c['avg_f'],
            avg_monetary=c['avg_m'],
            recommendation_channel=c['channel'],
            recommendation_keywords=c['keywords'],
            recommendation_message=c['message'],
            scatter_data_json=json.dumps(c['scatter'], ensure_ascii=False),
        ))
    CustomerCluster.objects.bulk_create(cluster_objs)

    # 차트 데이터 직렬화
    pie_data = [{'name': c['name'], 'value': c['count']} for c in result['clusters']]
    scatter_series = []
    for c in result['clusters']:
        scatter_series.append({
            'name': c['name'],
            'color': c['badge_color'],
            'data': [[p['r'], p['f'], p['m'], p['name']] for p in c['scatter']],
        })

    context = {
        'session': session,
        'clusters': result['clusters'],
        'pie_data_json': json.dumps(pie_data, ensure_ascii=False),
        'scatter_series_json': json.dumps(scatter_series, ensure_ascii=False),
    }
    return render(request, 'clustering/analysis_result.html', context)


@login_required
def session_result_view(request, session_id):
    from django.shortcuts import get_object_or_404
    from .services import CLUSTER_RULES

    session = get_object_or_404(AnalysisSession, id=session_id, user=request.user)
    clusters_qs = session.clusters.all()

    clusters = []
    pie_data = []
    scatter_series = []

    for cl in clusters_qs:
        rule = CLUSTER_RULES.get(cl.cluster_label, CLUSTER_RULES['general'])
        scatter_points = json.loads(cl.scatter_data_json)

        c = {
            'label':       cl.cluster_label,
            'name':        rule['name'],
            'description': rule['description'],
            'badge_color': rule['badge_color'],
            'count':       cl.customer_count,
            'avg_r':       cl.avg_recency,
            'avg_f':       cl.avg_frequency,
            'avg_m':       cl.avg_monetary,
            'channel':     cl.recommendation_channel,
            'keywords':    cl.recommendation_keywords,
            'message':     cl.recommendation_message,
            'scatter':     scatter_points,
        }
        clusters.append(c)
        pie_data.append({'name': rule['name'], 'value': cl.customer_count})
        scatter_series.append({
            'name':  rule['name'],
            'color': rule['badge_color'],
            'data':  [[p['r'], p['f'], p['m'], p['name']] for p in scatter_points],
        })

    context = {
        'session':           session,
        'clusters':          clusters,
        'pie_data_json':     json.dumps(pie_data, ensure_ascii=False),
        'scatter_series_json': json.dumps(scatter_series, ensure_ascii=False),
    }
    return render(request, 'clustering/analysis_result.html', context)


@login_required
def sample_csv_view(request):
    content = generate_sample_csv()
    response = HttpResponse(content, content_type='text/csv; charset=utf-8-sig')
    response['Content-Disposition'] = 'attachment; filename="sample_customers.csv"'
    return response
