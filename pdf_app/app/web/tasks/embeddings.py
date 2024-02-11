from celery import shared_task

from pdf_app.app.web.db.models import Pdf
from pdf_app.app.web.files import download
from pdf_app.app.chat import create_embeddings_for_pdf


@shared_task()
def process_document(pdf_id: int):
    pdf = Pdf.find_by(id=pdf_id)
    with download(pdf.id) as pdf_path:
        create_embeddings_for_pdf(pdf.id, pdf_path)
