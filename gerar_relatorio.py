#!/usr/bin/env python3
"""Gera relatório técnico em PDF a partir de métricas e evidências obtidas."""
from pathlib import Path
import json, csv, math
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT=Path(__file__).resolve().parent
stats=json.loads((ROOT/'dataset_stats.json').read_text())
val=json.loads((ROOT/'metricas_validacao.json').read_text())
test=json.loads((ROOT/'metricas_teste.json').read_text())
with (ROOT/'runs/apple_banana_yolov8n/results.csv').open() as f:
    epochs=list(csv.DictReader(f))
actual_epochs=len(epochs)
with (ROOT/'deteccoes.csv').open(encoding='utf-8') as f:
    detections=list(csv.DictReader(f))
classes={}
for d in detections: classes[d['classe']]=classes.get(d['classe'],0)+1
images=sorted((ROOT/'evidencias/deteccoes_imagens_ineditas').glob('*'))
test_image_names={
    p.name for p in (ROOT/'dataset/images/test').iterdir()
    if p.suffix.lower() in {'.jpg','.jpeg','.png'}
}
images=[p for p in images if p.suffix.lower() in {'.jpg','.jpeg','.png'}
        and p.name in test_image_names]
missing_images=test_image_names-{p.name for p in images}
if missing_images:
    raise FileNotFoundError(
        'Faltam imagens de evidência do holdout atual: '
        + ', '.join(sorted(missing_images))
    )

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCenter',parent=styles['Title'],alignment=TA_CENTER,textColor=colors.HexColor('#12334a'),fontSize=22,leading=27,spaceAfter=16))
styles.add(ParagraphStyle(name='Sub',parent=styles['Normal'],alignment=TA_CENTER,textColor=colors.HexColor('#456477'),fontSize=11,leading=16))
styles.add(ParagraphStyle(name='H1x',parent=styles['Heading1'],textColor=colors.HexColor('#136b78'),spaceBefore=12,spaceAfter=7))
styles.add(ParagraphStyle(name='Bodyx',parent=styles['BodyText'],fontSize=9.5,leading=14,spaceAfter=7))
styles.add(ParagraphStyle(name='Smallx',parent=styles['BodyText'],fontSize=8,leading=11,textColor=colors.HexColor('#4b5563')))

def fmt(k):
    v=float(k)
    return f'{v:.3f}' if math.isfinite(v) else 'não disponível'

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#607482'))
    canvas.drawString(0.65*inch,0.42*inch,'Atividade prática final — Inteligência Artificial — YOLO')
    canvas.drawRightString(7.85*inch,0.42*inch,f'{doc.page}')
    canvas.restoreState()

out=ROOT/'relatorio.pdf'
doc=SimpleDocTemplate(str(out),pagesize=letter,rightMargin=.65*inch,leftMargin=.65*inch,topMargin=.65*inch,bottomMargin=.65*inch,title='Detecção de maçãs e bananas com YOLO')
story=[]
story += [Spacer(1,.35*inch),Paragraph('DETECÇÃO DE MAÇÃS E BANANAS',styles['TitleCenter']),Paragraph('Treinamento e avaliação de um detector de objetos com YOLOv8',styles['Sub']),Spacer(1,.28*inch)]
info=[['Disciplina','Inteligência Artificial'],['Estudante','[INFORMAÇÃO QUE PRECISO FORNECER]'],['Modelo','YOLOv8n pré-treinado, fine-tuning, CPU'],['Treinamento',f'{actual_epochs} épocas registradas; imagens 320 × 320'],['Classes','apple (maçã) e banana']]
t=Table(info,colWidths=[1.45*inch,5.2*inch]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),colors.HexColor('#e8f1f4')),('TEXTCOLOR',(0,0),(0,-1),colors.HexColor('#12334a')),('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#c5d4da')),('FONTSIZE',(0,0),(-1,-1),9),('PADDING',(0,0),(-1,-1),7)])); story += [t,Spacer(1,18)]
story += [Paragraph('1. Problema e método',styles['H1x']),Paragraph('O projeto detecta caixas delimitadoras ao redor de maçãs e bananas em imagens. Foi usado o dataset público Fruits and Vegetables Detection, derivado de LVIS e distribuído em formato YOLO. Do conjunto de origem foram filtradas somente as duas classes-alvo. As IDs originais (apple=1 e banana=6) foram convertidas para 0 e 1; outras classes foram descartadas. A inicialização utilizou pesos pré-treinados YOLOv8n e ajuste fino supervisionado.',styles['Bodyx'])]
rows=[['Partição','Imagens','Caixas','Maçã','Banana']]
for split in ['train','val','test']:
 s=stats[split]; label={'train':'Treino','val':'Validação','test':'Imagens inéditas'}[split]
 rows.append([label,str(s['images']),str(s['boxes']),str(s['apple_boxes']),str(s['banana_boxes'])])
t=Table(rows,colWidths=[1.55*inch,.85*inch,.85*inch,1.0*inch,1.0*inch]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf0')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#b8c9cf')),('ALIGN',(1,1),(-1,-1),'CENTER'),('FONTSIZE',(0,0),(-1,-1),8.5),('PADDING',(0,0),(-1,-1),6)])); story += [t,Spacer(1,8),Paragraph(f"As {stats['test']['images']} imagens inéditas foram reservadas separadamente, não usadas no treino ou validação deste projeto. O recorte tem desbalanceamento entre caixas anotadas; por exemplo, no treino há {stats['train']['apple_boxes']} caixas de maçã e {stats['train']['banana_boxes']} de banana, condição que pode afetar o desempenho por classe.",styles['Bodyx'])]
story += [Paragraph('2. Treinamento e métricas',styles['H1x']),Paragraph(f"O treinamento configurou 20 épocas, com resolução de 320 pixels, batch 4 e execução em CPU. As métricas a seguir são calculadas pelo Ultralytics a partir das detecções nas partições indicadas. Precision é a fração das detecções previstas que está correta; Recall mede a fração dos objetos anotados encontrados; mAP50 é a média da precisão média com IoU de 0,50; mAP50–95 calcula essa média em limiares IoU de 0,50 a 0,95.",styles['Bodyx'])]
rows=[['Métrica',f"Validação ({stats['val']['images']} imagens)",f"Holdout ({test['holdout_images']} imagens)"],['Precision',fmt(val['precision']),fmt(test['precision'])],['Recall',fmt(val['recall']),fmt(test['recall'])],['mAP50',fmt(val['mAP50']),fmt(test['mAP50'])],['mAP50–95',fmt(val['mAP50_95']),fmt(test['mAP50_95'])]]
t=Table(rows,colWidths=[1.7*inch,2.0*inch,2.0*inch]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf0')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#b8c9cf')),('ALIGN',(1,1),(-1,-1),'CENTER'),('FONTSIZE',(0,0),(-1,-1),9),('PADDING',(0,0),(-1,-1),7)]));story += [t,Spacer(1,5),Paragraph('Os resultados devem ser interpretados com cautela: o conjunto de validação e o holdout são pequenos, e há forte desequilíbrio entre as classes. Métricas não são uma garantia de generalização para qualquer ambiente ou imagem.',styles['Smallx'])]
story += [PageBreak(),Paragraph('3. Respostas às questões do enunciado',styles['H1x'])]
answers=[
('Questão 1 — Qual problema foi escolhido e por quê?','Detecção de maçãs e bananas em imagens, um problema visual simples e observável que permite demonstrar detecção de múltiplas classes e localização por caixas.'),
('Questão 2 — Quantas imagens e classes foram utilizadas? Como as imagens foram obtidas?',f"Foram selecionadas {stats['train']['images']} imagens de treino, {stats['val']['images']} de validação e {stats['test']['images']} inéditas de demonstração, totalizando {sum(stats[split]['images'] for split in ['train','val','test'])} imagens. As imagens e caixas foram obtidas do dataset público Fruits and Vegetables Detection/LVIS; somente apple e banana foram mantidas e remapeadas para as IDs 0 e 1."),
('Questão 3 — Qual modelo YOLO e quantas épocas?','Foi utilizado YOLOv8n com pesos pré-treinados como inicialização. A execução registrou '+str(actual_epochs)+' épocas completas, usando imgsz=320 e CPU.'),
('Questão 4 — Quais foram Precision, Recall e mAP? O que significam?','Na validação: Precision='+fmt(val['precision'])+', Recall='+fmt(val['recall'])+', mAP50='+fmt(val['mAP50'])+' e mAP50–95='+fmt(val['mAP50_95'])+'. No holdout inédito: Precision='+fmt(test['precision'])+', Recall='+fmt(test['recall'])+', mAP50='+fmt(test['mAP50'])+' e mAP50–95='+fmt(test['mAP50_95'])+'. Precision relaciona acertos com todas as detecções previstas; Recall relaciona acertos com objetos verdadeiros; mAP resume a precisão média das classes segundo IoU.'),
('Questão 5 — O modelo reconheceu os objetos nas imagens inéditas?','O modelo executou a detecção nas '+str(stats['test']['images'])+' imagens, gerando '+str(len(detections))+' caixas com confiança mínima 0,20. Porém, a inspeção visual revelou múltiplas caixas sobrepostas e classes atribuídas a objetos que não são maçã ou banana; portanto, o reconhecimento foi apenas parcial e não confiável. As métricas por classe e as imagens anotadas permitem observar exemplos dessas falhas. As pontuações de confiança não equivalem, isoladamente, à probabilidade de acerto.'),
('Questão 6 — Quais dificuldades e melhorias são possíveis?','As principais limitações são o conjunto pequeno, o desbalanceamento das caixas (especialmente menos exemplos de maçãs), execução em CPU e variação de contexto, escala e iluminação. Melhorias incluem coletar mais imagens próprias e diversas, equilibrar as classes, revisar anotações, aumentar épocas/resolução quando houver recursos, ajustar hiperparâmetros e avaliar em um conjunto de teste maior e externo.')]
for q,a in answers: story.append(KeepTogether([Paragraph(q,styles['Heading3']),Paragraph(a,styles['Bodyx'])]))
story += [Paragraph('4. Evidências de inferência',styles['H1x']),Paragraph('A inferência usa limiar de confiança 0,20 e NMS com IoU 0,45. Abaixo estão imagens inéditas com previsões desenhadas pelo Ultralytics. A inspeção da primeira imagem mostra caixas repetidas e banana atribuída a partes de objetos vegetais que não são banana, evidenciando falsos positivos e o limite do modelo neste experimento. As saídas integrais e a tabela de detecções acompanham os scripts no pacote do projeto.',styles['Bodyx'])]
for img in images[:3]:
    story.append(Image(str(img),width=5.8*inch,height=3.7*inch,kind='proportional'))
    story.append(Paragraph(img.name,styles['Smallx']))
    story.append(Spacer(1,8))
story += [Paragraph('5. Arquivos do projeto',styles['H1x']),Paragraph('O pacote contém o dataset recortado em formato YOLO, o arquivo data.yaml, o checkpoint best.pt, os scripts de preparação/treinamento/validação/inferência, as métricas JSON/CSV e as evidências visuais. O nome do estudante precisa ser inserido no campo sinalizado antes da entrega final.',styles['Bodyx']),Paragraph('Fontes: dataset Fruits and Vegetables Detection — https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset; documentação Ultralytics — https://docs.ultralytics.com/modes/train/. A licença fornecida no repositório acompanha o pacote.',styles['Smallx'])]
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
