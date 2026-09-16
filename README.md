# FlyH1-Drone

Drone autônomo controlado por neurônios H1 da mosca (*Calliphora vicina*).

## O que é

O neurônio H1 é um neurônio tangencial da lobula plate que codifica velocidade de optic flow em taxa de spike. Moscas usam esses neurônios pra navegação em vôo — detecção de obstáculos, estabilização, controle de velocidade.

Este projeto implementa:
1. **Modelo LIF do H1** baseado em Borst & Haag (1996) — simula o neurônio com parâmetros reais
2. **Simulador de drone** onde 3 neurônios H1 (esquerda, direita, frente) controlam um quadrotor navegando entre obstáculos

## Como rodar

```bash
# Instalar dependências
pip install -r requirements.txt

# 1. Ver o neurônio H1 isolado
python h1_neuron.py

# 2. Rodar o drone com visualização 3D
python drone_sim.py

# 3. Rodar headless (sem janela)
python drone_sim.py --headless
```

## Arquitetura

```
câmera simulada → optic flow → H1 neurons (3x) → spike rates → controlador → drone
                                  │
                                  ├── H1_left:  flow esquerdo → vira direita
                                  ├── H1_right: flow direito  → vira esquerda
                                  └── H1_front: looming       → freia + sobe
```

## Referências

- Borst, A. & Haag, J. (1996). "The intrinsic electrophysiology of fly lobula plate tangential cells"
- de Ruyter van Steveninck et al. (1997). "Reproducibility and variability in neural spike trains"
- Connectome: FlyWire/Princeton (2023) — mapeamento completo do cérebro da *Drosophila*
