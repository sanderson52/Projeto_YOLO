# Fontes e proveniência

1. Enunciado da atividade: PDF fornecido pelo estudante, `Atividade_Pratica_Final_IA_YOLO.pdf`.
2. Dataset Fruits and Vegetables Detection, derivado de LVIS e publicado no formato YOLO: https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset. O repositório informa 8.221 imagens de origem (6.721 treino, 1.500 validação e 180 amostras de teste anotadas manualmente). Este projeto usa um recorte de 65 imagens e filtra as classes apple e banana.
3. Documentação Ultralytics para treinamento: https://docs.ultralytics.com/modes/train/.
4. O arquivo de licença fornecido pelo repositório está em `LICENSE_dataset.txt`. O dataset publicado deriva de LVIS; consulte também as condições de atribuição/licença indicadas pelo projeto de origem antes de redistribuir as imagens.

As métricas registradas pelo experimento estão em `metricas_validacao.json` e `metricas_teste.json`; as contagens de imagens e caixas foram calculadas em `dataset_stats.json`.
