"""
Drone controlado pelo neurônio H1 da mosca
==========================================

Pipeline:
  câmera simulada → optic flow → H1 (spike rate) → controlador → drone corrige trajetória

O drone voa num ambiente PyBullet com obstáculos. A H1 detecta movimento no campo
visual e faz o drone desviar — exatamente como a mosca faz.

Uso:
    python drone_sim.py

Requisitos:
    pip install -r requirements.txt
"""

import numpy as np
import time
import sys
import os

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("PyBullet não instalado. Rode: pip install pybullet")
    print("Ou: pip install -r requirements.txt")
    sys.exit(1)

from h1_neuron import H1Neuron


# =========================================================================
# OPTIC FLOW SIMULATOR (substitui câmera real)
# =========================================================================
class OpticFlowSimulator:
    """
    Calcula optic flow a partir da posição/velocidade do drone e obstáculos.
    Na mosca real, isso é feito por fotorreceptores + neurônios da medula.
    Aqui simplificamos: projetamos velocidade relativa dos obstáculos no frame do drone.
    """

    def __init__(self):
        self.obstacles = []

    def add_obstacle(self, position, radius=0.5):
        self.obstacles.append({'pos': np.array(position), 'radius': radius})

    def compute_flow(self, drone_pos, drone_vel, drone_yaw):
        """
        Retorna optic flow percebido pelo drone.

        Returns:
            dict com flow_left, flow_right, flow_front (°/s)
            Esses são os 3 canais que alimentam a H1.

        v3 — corrige saturação dos 3 neurônios:
        - Cutoff de distância: obstáculos >4m são ignorados (background)
        - Falloff suave: contribuição cai com 1/dist³ (não 1/dist²)
        - Multiplicadores calibrados pra que neurônios fiquem em 0-200 Hz
          normalmente, e só se aproximem de 500 Hz com obstáculo a <1m
        """
        flow = {'left': 0.0, 'right': 0.0, 'front': 0.0}

        if len(self.obstacles) == 0:
            return flow

        cos_yaw = np.cos(drone_yaw)
        sin_yaw = np.sin(drone_yaw)
        forward = np.array([cos_yaw, sin_yaw])

        max_range = 5.0  # v4: aumentado de 4m pra 5m — mais tempo de reação

        for obs in self.obstacles:
            # Vetor 2D do drone pro obstáculo
            rel = obs['pos'][:2] - drone_pos[:2]
            dist = max(np.linalg.norm(rel), 0.1)

            # Ignorar obstáculos distantes
            if dist > max_range:
                continue

            # Falloff suave: 1.0 a dist=0, 0.0 a dist=max_range
            range_factor = max(0.0, 1.0 - (dist / max_range) ** 2)

            # ---- LATERAL: velocidade angular correta ----
            # ω = (vx * dy - vy * dx) / dist²
            cross = drone_vel[0] * rel[1] - drone_vel[1] * rel[0]
            angular_vel_rads = cross / (dist * dist)

            # Escala pro range do H1 — calibrado pra não saturar longe
            # Tamanho aparente: obs['radius'] / dist (radianos)
            apparent_size = obs['radius'] / dist
            angular_vel_deg = np.degrees(abs(angular_vel_rads)) * (1.0 + apparent_size * 3.0) * range_factor * 3.0

            if cross > 0:
                flow['left'] += angular_vel_deg
            else:
                flow['right'] += angular_vel_deg

            # ---- LOOMING: cone frontal de 60° ----
            rel_unit = rel / dist
            in_front = np.dot(rel_unit, forward) > 0.5  # cos(60°) = 0.5

            if in_front:
                approach_vel = np.dot(drone_vel[:2], rel_unit)
                if approach_vel > 0:  # se aproximando
                    # Taxa de expansão: r * v / d²
                    looming = approach_vel * obs['radius'] / (dist * dist)
                    flow['front'] += np.degrees(looming) * range_factor * 15.0

        return flow


# =========================================================================
# H1 CONTROLLER: 3 neurônios H1 → comandos de vôo
# =========================================================================
class H1Controller:
    """
    Usa 3 neurônios H1 (esquerda, direita, frente) pra controlar o drone.
    Baseado na organização bilateral da lobula plate da mosca.

    - H1_left detecta flow à esquerda → vira pra direita
    - H1_right detecta flow à direita → vira pra esquerda
    - H1_front detecta looming → freia / sobe
    """

    def __init__(self):
        self.h1_left = H1Neuron()
        self.h1_right = H1Neuron()
        self.h1_front = H1Neuron()

        # Ganhos v5 — calibrados pra yaw_rate em rad/s (aplicado * sim_dt no loop)
        # Com rate=100 Hz → yaw_rate = 0.3 rad/s → 0.3/240 ≈ 0.001 rad/step
        self.yaw_gain = 0.003
        self.brake_gain = 0.003
        self.altitude_gain = 0.001

        # Reflexo de escape: quando looming é alto, drone SEMPRE escapa
        # (independente de simetria — simetria só define a DIREÇÃO)
        self.escape_threshold = 100.0   # Hz: looming acima disso ativa escape
        self.escape_gain = 0.8          # ganho forte do escape (rad/s)

    def step(self, flow: dict, dt: float = 1e-4) -> dict:
        """
        Recebe optic flow, retorna comandos de controle.

        v4: adiciona reflexo de escape (dodge response).
        Quando looming é alto e os sinais laterais são simétricos (obstáculo
        quase em frente), o drone escolhe o lado com menos flow e vira forte.
        Baseado no escape response da mosca (Card & Dickinson, 2008).

        Returns:
            dict com yaw_rate, throttle_adjust, pitch_adjust, rates
        """
        r_left = self.h1_left.step(flow['left'], dt)
        r_right = self.h1_right.step(flow['right'], dt)
        r_front = self.h1_front.step(flow['front'], dt)

        rate_left = r_left['spike_rate']
        rate_right = r_right['spike_rate']
        rate_front = r_front['spike_rate']

        # --- Controle diferencial normal ---
        # obstacle à esquerda → rate_left alto → yaw negativo → vira à DIREITA (away)
        yaw_rate = (rate_right - rate_left) * self.yaw_gain

        # --- Reflexo de escape lateral ---
        # Ativado quando looming alto — SEMPRE.
        # Simetria só define a direção (o lado com menos flow = mais espaço).
        if rate_front > self.escape_threshold:
            # Escolhe o lado com MENOS flow (mais espaço livre)
            # Se ambos zero (obstáculo direto em frente): escolhe direita por default
            if rate_left <= rate_right:
                escape_dir = 1.0   # vira pra direita
            else:
                escape_dir = -1.0  # vira pra esquerda

            # Intensidade proporcional ao excesso acima do threshold
            escape_intensity = (rate_front - self.escape_threshold) / 400.0
            escape_intensity = np.clip(escape_intensity, 0.0, 1.0)
            yaw_rate += escape_dir * escape_intensity * self.escape_gain

        # --- Looming frontal → freia ---
        pitch_adjust = -rate_front * self.brake_gain
        throttle_adjust = rate_front * self.altitude_gain

        return {
            'yaw_rate': np.clip(yaw_rate, -2.0, 2.0),
            'pitch_adjust': np.clip(pitch_adjust, -0.6, 0.0),
            'throttle_adjust': np.clip(throttle_adjust, 0.0, 0.5),
            'rates': {
                'left': rate_left,
                'right': rate_right,
                'front': rate_front,
            }
        }


# =========================================================================
# SIMULAÇÃO PRINCIPAL
# =========================================================================
def run_simulation(gui=True, duration=30.0):
    """
    Roda a simulação completa.

    Args:
        gui: True pra abrir janela 3D, False pra headless
        duration: duração em segundos
    """
    print("=" * 60)
    print("DRONE H1 — Controle neurobiológico inspirado em Calliphora")
    print("=" * 60)

    # --- PyBullet setup ---
    if gui:
        physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    else:
        physics_client = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    # Chão
    plane_id = p.loadURDF("plane.urdf")

    # Drone (representado por um cubo simples — sem URDF complexo)
    drone_start_pos = [0, 0, 1.5]
    drone_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.05])
    vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.05],
                                     rgbaColor=[0.2, 0.8, 0.2, 1.0])
    drone_id = p.createMultiBody(
        baseMass=0.5,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=drone_start_pos,
        baseOrientation=drone_start_orn
    )

    # Desabilitar dinâmica padrão — controlamos diretamente
    p.changeDynamics(drone_id, -1, linearDamping=0.9, angularDamping=0.9)

    # Obstáculos (cilindros vermelhos)
    obstacles_pos = [
        [3, 0.5, 1.0],
        [5, -1.0, 1.0],
        [7, 0.8, 1.0],
        [9, -0.3, 1.0],
        [4, 2.0, 1.0],
        [6, -2.0, 1.0],
        [8, 1.5, 1.0],
    ]

    flow_sim = OpticFlowSimulator()
    for obs_pos in obstacles_pos:
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=2.0)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=2.0,
                                   rgbaColor=[0.8, 0.2, 0.2, 0.8])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                         baseVisualShapeIndex=vis, basePosition=obs_pos)
        flow_sim.add_obstacle(obs_pos, radius=0.3)

    # Meta (esfera azul no final do corredor)
    goal_pos = [11, 0, 1.5]
    goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.4,
                                    rgbaColor=[0.2, 0.4, 1.0, 0.6])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_vis,
                     basePosition=goal_pos)

    # --- Controller ---
    controller = H1Controller()

    # --- Simulação ---
    sim_dt = 1.0 / 240.0  # PyBullet default
    h1_dt = 1e-4           # H1 opera em 0.1ms steps
    h1_steps_per_sim = int(sim_dt / h1_dt)

    # Velocidade base do drone (avança pra frente constantemente)
    base_speed = 0.7  # m/s — v3: entre v1 (0.8) e v2 (0.55)

    # Câmera tracking
    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=4, cameraYaw=-30, cameraPitch=-30,
            cameraTargetPosition=[3, 0, 1]
        )

    print(f"\nSimulação iniciada — {duration}s, {len(obstacles_pos)} obstáculos")
    print(f"Base speed: {base_speed} m/s")
    print(f"H1 neurons: 3 (left/right/front)")
    print(f"Meta: chegar em x={goal_pos[0]}m sem colidir\n")

    log = {
        'time': [], 'pos_x': [], 'pos_y': [], 'pos_z': [],
        'rate_left': [], 'rate_right': [], 'rate_front': [],
        'yaw': [], 'collisions': 0
    }

    sim_time = 0.0
    step_count = 0

    while sim_time < duration:
        # Estado do drone
        pos, orn = p.getBasePositionAndOrientation(drone_id)
        vel, ang_vel = p.getBaseVelocity(drone_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        pos = np.array(pos)
        vel = np.array(vel)

        # Checar colisão
        contacts = p.getContactPoints(bodyA=drone_id)
        if len(contacts) > 0:
            for c in contacts:
                if c[2] != plane_id:  # ignorar chão
                    log['collisions'] += 1
                    if step_count % 240 == 0:
                        print(f"  [!] Colisão em t={sim_time:.1f}s pos=({pos[0]:.1f}, {pos[1]:.1f})")

        # Checar se chegou na meta
        if pos[0] >= goal_pos[0]:
            print(f"\n✓ META ALCANÇADA em t={sim_time:.1f}s!")
            print(f"  Colisões: {log['collisions']}")
            break

        # Computar optic flow
        flow = flow_sim.compute_flow(pos, vel, yaw)

        # Rodar H1 controller (múltiplos passos por step de física)
        cmd = None
        for _ in range(h1_steps_per_sim):
            cmd = controller.step(flow, h1_dt)

        # Aplicar controle
        # CORREÇÃO: multiplicar yaw_rate por sim_dt — sem isso vira ~86°/step
        target_yaw = yaw + cmd['yaw_rate'] * sim_dt
        forward_speed = max(0.1, base_speed + cmd['pitch_adjust'])
        target_vz = cmd['throttle_adjust']

        # Velocidade no frame world
        vx = forward_speed * np.cos(target_yaw)
        vy = forward_speed * np.sin(target_yaw)
        vz = (1.5 - pos[2]) * 2.0 + target_vz  # PD pra altitude + ajuste H1

        p.resetBaseVelocity(drone_id, [vx, vy, vz], [0, 0, cmd['yaw_rate'] * 5])

        # Log
        if step_count % 24 == 0:  # a cada ~0.1s
            log['time'].append(sim_time)
            log['pos_x'].append(pos[0])
            log['pos_y'].append(pos[1])
            log['pos_z'].append(pos[2])
            log['rate_left'].append(cmd['rates']['left'])
            log['rate_right'].append(cmd['rates']['right'])
            log['rate_front'].append(cmd['rates']['front'])
            log['yaw'].append(np.degrees(yaw))

        # Print periódico
        if step_count % 480 == 0 and step_count > 0:
            print(f"  t={sim_time:.1f}s  x={pos[0]:.1f}  y={pos[1]:.1f}  "
                  f"H1: L={cmd['rates']['left']:.0f} R={cmd['rates']['right']:.0f} "
                  f"F={cmd['rates']['front']:.0f} Hz")

        p.stepSimulation()
        sim_time += sim_dt
        step_count += 1

        if gui:
            # Tracking camera
            if step_count % 10 == 0:
                p.resetDebugVisualizerCamera(
                    cameraDistance=4, cameraYaw=-30, cameraPitch=-30,
                    cameraTargetPosition=pos.tolist()
                )
            time.sleep(sim_dt * 0.5)  # slow down pra visualizar

    # --- Plot dos resultados ---
    print("\nGerando gráficos...")

    import matplotlib
    try:
        import subprocess
        subprocess.run(['xdpyinfo'], capture_output=True, check=True)
    except Exception:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#0a0a0a')

    colors = {'bg': '#0a0a0a', 'grid': '#1a1a2e', 'text': '#e0e0e0'}

    # Panel 1: Trajetória top-down
    ax1 = axes[0]
    ax1.set_facecolor(colors['bg'])
    ax1.plot(log['pos_x'], log['pos_y'], color='#00ff88', linewidth=2, label='Drone')
    for obs in obstacles_pos:
        circle = plt.Circle((obs[0], obs[1]), 0.3, color='#ff3366', alpha=0.7)
        ax1.add_patch(circle)
    ax1.plot(goal_pos[0], goal_pos[1], 'o', color='#00d4ff', markersize=15, label='Meta')
    ax1.set_xlabel('X (m)', color=colors['text'])
    ax1.set_ylabel('Y (m)', color=colors['text'])
    ax1.set_title('Trajetória do Drone (vista de cima)', color=colors['text'], fontweight='bold')
    ax1.set_aspect('equal')
    ax1.legend(facecolor=colors['bg'], edgecolor=colors['grid'], labelcolor=colors['text'])
    ax1.grid(True, alpha=0.15, color=colors['grid'])
    ax1.tick_params(colors=colors['text'])
    for spine in ax1.spines.values():
        spine.set_color(colors['grid'])

    # Panel 2: Spike rates dos 3 H1
    ax2 = axes[1]
    ax2.set_facecolor(colors['bg'])
    ax2.plot(log['time'], log['rate_left'], color='#ff6b35', linewidth=1, label='H1 Left', alpha=0.8)
    ax2.plot(log['time'], log['rate_right'], color='#00d4ff', linewidth=1, label='H1 Right', alpha=0.8)
    ax2.plot(log['time'], log['rate_front'], color='#ff3366', linewidth=1, label='H1 Front (looming)', alpha=0.8)
    ax2.set_xlabel('Tempo (s)', color=colors['text'])
    ax2.set_ylabel('Spike Rate (Hz)', color=colors['text'])
    ax2.set_title('Atividade dos Neurônios H1', color=colors['text'], fontweight='bold')
    ax2.legend(facecolor=colors['bg'], edgecolor=colors['grid'], labelcolor=colors['text'])
    ax2.grid(True, alpha=0.15, color=colors['grid'])
    ax2.tick_params(colors=colors['text'])
    for spine in ax2.spines.values():
        spine.set_color(colors['grid'])

    # Panel 3: Yaw (direção)
    ax3 = axes[2]
    ax3.set_facecolor(colors['bg'])
    ax3.plot(log['time'], log['yaw'], color='#00ff88', linewidth=1.5)
    ax3.set_xlabel('Tempo (s)', color=colors['text'])
    ax3.set_ylabel('Yaw (°)', color=colors['text'])
    ax3.set_title('Direção do Drone', color=colors['text'], fontweight='bold')
    ax3.grid(True, alpha=0.15, color=colors['grid'])
    ax3.tick_params(colors=colors['text'])
    for spine in ax3.spines.values():
        spine.set_color(colors['grid'])

    plt.tight_layout()
    plt.savefig('drone_h1_results.png', dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    print(f"✓ Resultados salvos em drone_h1_results.png")

    p.disconnect()

    print(f"\n{'='*60}")
    print(f"RESUMO")
    print(f"{'='*60}")
    print(f"Tempo total: {sim_time:.1f}s")
    print(f"Colisões: {log['collisions']}")
    print(f"Distância percorrida: {log['pos_x'][-1] if log['pos_x'] else 0:.1f}m")
    print(f"Desvio lateral max: {max(abs(np.array(log['pos_y']))) if log['pos_y'] else 0:.2f}m")

    return log


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Drone H1 — Controle neurobiológico')
    parser.add_argument('--headless', action='store_true', help='Rodar sem janela 3D')
    parser.add_argument('--duration', type=float, default=30.0, help='Duração da simulação (s)')
    args = parser.parse_args()

    run_simulation(gui=not args.headless, duration=args.duration)
