# Plano de Implementação do Sistema de Detecção de Anomalias no Tráfego Urbano

## 1. Estrutura dos Módulos

- **Pré-processamento**
    - Leitura e estabilização de vídeo
    - Geração de máscaras de interesse
    - Detecção de veículos (ex: YOLO, Mask R-CNN)
    - Aumento de dados com WGAN-GP

- **Rastreamento**
    - Rastreamento por caixas delimitadoras usando IOU
    - Geração de tubos temporais (sequências de caixas)
    - Fusão inter-tubo para evitar duplicidade de dados

- **Pós-processamento**
    - Classificação de anomalias
    - Refinamento de bordas espaço-temporais
    - Otimização para redução de falsos positivos