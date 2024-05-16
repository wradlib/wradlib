from django.urls import path
from . import views

urlpatterns = [
    path("", views.my_view, name="my_view-path"),
    path("all-senarios/", views.list_location, name="all-senarios"),
    path("detail/<int:pk>/", views.detail, name="detail"),
    path("delete/<int:pk>/", views.delete_simulation, name="delete"),
    path("sites/", views.list_site, name="list_site"),
    path("sites/<str:name>/", views.detail_site, name="detail_site"),
    path("update/<int:pk>/", views.update, name="update"),
    path("report-pdf/", views.radar_render_pdf_view, name="pdf-path-2"),
    path("report-pdf/<str:name>/", views.site_render_pdf_view, name="pdf-site"),
]
