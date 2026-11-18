# Reconhecimento de maçãs e bananas com YOLO

Projeto acadêmico de detecção de objetos com caixas delimitadoras, baseado em um recorte do dataset público Fruits and Vegetables Detection (LVIS, formato YOLO). As classes escolhidas são `apple` e `banana`. Os scripts selecionam imagens com rótulos válidos, convertem as classes originais (apple=1, banana=6) para os IDs 0 e 1 e criam um holdout separado.

## Pré-requisitos e execução

Ambiente usado nesta execução: Linux x86_64, Python 3.12.3. O treinamento detecta automaticamente CPU/GPU; neste ambiente foi planejado para CPU por não haver CUDA disponível.

```bash
python3 -m pip install ultralytics
python3 baixar_dataset.py
python3 preparar_dataset.py
python3 treinar_validar.py
python3 detectar_ineditas.py
python3 avaliar_teste.py
python3 gerar_relatorio.py
```

O primeiro passo baixa os arquivos-fonte do dataset (aproximadamente 1,4 GB); eles não são incluídos no ZIP. O recorte já preparado, com 65 imagens e rótulos, está incluído. O `data.yaml` usa caminho relativo para continuar funcionando após extrair o pacote em outra pasta.

O treinamento usa `yolov8n.pt` pré-treinado, 20 épocas configuradas, `imgsz=320`, batch 4, CPU e semente fixa. O melhor checkpoint é copiado para `modelo/best.pt`. Os detalhes reais (épocas realizadas e métricas) são registrados em `metricas_validacao.json`, e as inferências em `deteccoes.csv`, `deteccoes.json` e `evidencias/`.

## Partições e anotações

`preparar_dataset.py` escolhe 50 imagens para treino, 10 para validação e cinco imagens adicionais para demonstração. As imagens de demonstração são selecionadas de forma separada e seus nomes são excluídos das partições treino/validação deste projeto. Cada execução substitui os arquivos gerados nas partições, evitando que imagens de execuções anteriores sejam acumuladas. A execução imprime os números reais de imagens e caixas e os salva em `dataset_stats.json`. O YAML gerado fica em `dataset/data.yaml`.

As caixas estão no formato YOLO normalizado `classe x_centro y_centro largura altura`. Outras classes presentes nas anotações da origem são desconsideradas, pois a aplicação aprende apenas maçã e banana.

## Arquivos de entrega

- `dataset/`: recorte utilizado, rótulos e configuração.
- `modelo/best.pt`: melhores pesos do treinamento.
- `runs/`: logs, curvas e resultados do Ultralytics.
- `evidencias/`: imagens inéditas com caixas previstas.
- `relatorio.pdf`: relatório preenchido com valores obtidos na execução.
- Scripts `.py`: preparação, treinamento/validação e inferência.

Dataset de origem: https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset. Consulte `LICENSE_dataset.txt` e `fontes.md`.
