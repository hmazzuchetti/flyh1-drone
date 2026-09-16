"""
Modelo do neurônio H1 da mosca (Calliphora vicina)
===================================================

Baseado em:
- Borst & Haag (1996) — "The intrinsic electrophysiology of fly lobula plate tangential cells"
- de Ruyter van Steveninck et al. (1997) — "Reproducibility and variability in neural spike trains"

O H1 é um neurônio tangencial da lobula plate que responde a movimento horizontal
(optic flow). Codifica velocidade angular em taxa de spike.

Uso standalone:
    python h1_neuron.py

Uso como módulo:
    from h1_neuron import H1Neuron
    h1 = H1Neuron()
    spike_rate = h1.step(velocity=200.0, dt=0.001)
"""

import numpy as np


class H1Neuron:
    """Leaky Integrate-and-Fire model do neurônio H1."""

    # Parâmetros da literatura (Borst & Haag 1996)
    TAU_M = 10e-3        # constante de tempo da membrana (10ms)
    V_REST = -60e-3      # potencial de repouso (-60mV)
    V_THRESH = -45e-3    # limiar de disparo (-45mV)
    V_RESET = -65e-3     # potencial pós-spike (-65mV)
    R_M = 40e6           # resistência de membrana (40 MΩ)
    T_REF = 2e-3         # período refratário (2ms)
    GAIN = 0.25e-9       # nA por grau/s — reduzido 10x pra expandir range dinâmico
                         # Antes: 2.5e-9 saturava em ~340 Hz com 1°/s de flow
                         # Agora: range útil ~0-400 Hz ao longo de 0-100°/s
    NOISE_STD = 0.05e-9  # ruído sináptico — reduzido junto pro baseline ficar limpo

    def __init__(self):
        self.V = self.V_REST
        self.last_spike_time = -np.inf
        self.time = 0.0
        self.spike_history = []
        self._rate_window = 50e-3  # janela de 50ms pra calcular taxa

    def reset(self):
        """Reseta o estado do neurônio."""
        self.V = self.V_REST
        self.last_spike_time = -np.inf
        self.time = 0.0
        self.spike_history = []

    def step(self, velocity: float, dt: float = 1e-4) -> dict:
        """
        Avança um passo temporal.

        Args:
            velocity: velocidade angular do optic flow (°/s)
            dt: passo temporal em segundos (default 0.1ms)

        Returns:
            dict com:
                - spiked: bool
                - voltage: float (mV)
                - spike_rate: float (Hz) — taxa instantânea (janela deslizante)
                - time: float (s)
        """
        self.time += dt
        spiked = False

        # Período refratário
        if (self.time - self.last_spike_time) < self.T_REF:
            self.V = self.V_RESET
        else:
            # Corrente de input (retificada — H1 é direction-selective)
            I = max(0.0, velocity * self.GAIN + np.random.normal(0, self.NOISE_STD))

            # Equação LIF: tau_m * dV/dt = -(V - V_rest) + R_m * I
            dV = (-(self.V - self.V_REST) + self.R_M * I) * dt / self.TAU_M
            self.V += dV

            # Disparo
            if self.V >= self.V_THRESH:
                spiked = True
                self.spike_history.append(self.time)
                self.V = self.V_RESET
                self.last_spike_time = self.time

        # Taxa de disparo (janela deslizante)
        cutoff = self.time - self._rate_window
        recent = [s for s in self.spike_history if s > cutoff]
        spike_rate = len(recent) / self._rate_window

        # Limpar histórico antigo pra não acumular memória
        if len(self.spike_history) > 1000:
            self.spike_history = self.spike_history[-500:]

        return {
            'spiked': spiked,
            'voltage': self.V * 1000,  # em mV
            'spike_rate': spike_rate,
            'time': self.time,
        }

    def get_rate_for_velocity(self, velocity: float) -> float:
        """
        Calcula taxa de disparo analítica pra uma velocidade constante.
        Útil pra transfer function sem simular.
        """
        I = velocity * self.GAIN
        numerator = self.R_M * I + self.V_REST - self.V_RESET
        denominator = self.R_M * I + self.V_REST - self.V_THRESH
        if denominator > 0 and numerator > 0:
            return 1.0 / (self.T_REF + self.TAU_M * np.log(numerator / denominator))
        return 0.0


# =========================================================================
# DEMO: roda se executado diretamente
# =========================================================================
if __name__ == '__main__':
    import matplotlib
    import sys

    # Detecta se tem display disponível
    if sys.platform != 'win32':
        try:
            import subprocess
            subprocess.run(['xdpyinfo'], capture_output=True, check=True)
        except Exception:
            matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    print("Simulando neurônio H1...")

    h1 = H1Neuron()
    dt = 0.1e-3
    T = 3.0
    t = np.arange(0, T, dt)

    # Cenário: regimes de velocidade
    def optic_flow_scenario(time):
        if time < 0.5:
            return 5.0          # parada
        elif time < 1.0:
            return 50.0         # caminhando
        elif time < 1.5:
            return 200.0        # vôo normal
        elif time < 1.7:
            return 800.0        # saccade
        else:
            return 150 + 120 * np.sin(2 * np.pi * 3 * time)  # turbulência

    velocities = []
    voltages = []
    rates = []
    spike_times = []

    for ti in t:
        vel = optic_flow_scenario(ti)
        result = h1.step(vel, dt)
        velocities.append(vel)
        voltages.append(result['voltage'])
        rates.append(result['spike_rate'])
        if result['spiked']:
            spike_times.append(ti)

    spike_times = np.array(spike_times)
    print(f"Total de spikes: {len(spike_times)}")
    print(f"Taxa média: {len(spike_times)/T:.1f} Hz")

    # --- Plot ---
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor('#0a0a0a')
    gs = GridSpec(4, 1, height_ratios=[1, 1.2, 0.5, 1.2], hspace=0.35)

    colors = {
        'flow': '#00d4ff', 'voltage': '#ff6b35', 'spike': '#00ff88',
        'rate': '#ff3366', 'bg': '#0a0a0a', 'grid': '#1a1a2e', 'text': '#e0e0e0'
    }

    # Panel 1: Optic Flow
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(colors['bg'])
    ax1.plot(t, velocities, color=colors['flow'], linewidth=1.5, alpha=0.9)
    ax1.fill_between(t, 0, velocities, color=colors['flow'], alpha=0.15)
    ax1.set_ylabel('Velocidade angular\n(°/s)', color=colors['text'], fontsize=11)
    ax1.set_title('SIMULADOR H1 — Neurônio de Optic Flow da Mosca',
                  color=colors['text'], fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlim(0, T)
    ax1.tick_params(colors=colors['text'])
    ax1.grid(True, alpha=0.15, color=colors['grid'])
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax1.spines[spine].set_color(colors['grid'])

    # Anotações
    for x, label in [(0.25, 'PARADA'), (0.75, 'LENTO'), (1.25, 'VÔO'),
                     (1.6, 'SACCADE'), (2.35, 'TURBULÊNCIA')]:
        ax1.annotate(label, xy=(x, 0), xytext=(x, 850), fontsize=7,
                    color=colors['flow'], alpha=0.7, ha='center')

    # Panel 2: Membrane Voltage
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(colors['bg'])
    ax2.plot(t, voltages, color=colors['voltage'], linewidth=0.5, alpha=0.8)
    ax2.axhline(y=-45, color=colors['spike'], linewidth=0.8, linestyle='--', alpha=0.5, label='Limiar (-45 mV)')
    ax2.axhline(y=-60, color=colors['text'], linewidth=0.5, linestyle=':', alpha=0.3, label='Repouso (-60 mV)')
    ax2.set_ylabel('Potencial de\nmembrana (mV)', color=colors['text'], fontsize=11)
    ax2.set_xlim(0, T)
    ax2.tick_params(colors=colors['text'])
    ax2.grid(True, alpha=0.15, color=colors['grid'])
    ax2.legend(fontsize=8, facecolor=colors['bg'], edgecolor=colors['grid'],
               labelcolor=colors['text'], loc='upper right')
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax2.spines[spine].set_color(colors['grid'])

    # Panel 3: Spike Raster
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(colors['bg'])
    if len(spike_times) > 0:
        ax3.eventplot([spike_times], colors=[colors['spike']], linewidths=0.8, linelengths=0.8)
    ax3.set_ylabel('Spikes', color=colors['text'], fontsize=11)
    ax3.set_xlim(0, T)
    ax3.set_yticks([])
    ax3.tick_params(colors=colors['text'])
    for spine in ['top', 'right']:
        ax3.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax3.spines[spine].set_color(colors['grid'])

    # Panel 4: Firing Rate
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor(colors['bg'])
    ax4.plot(t, rates, color=colors['rate'], linewidth=1.5, alpha=0.9)
    ax4.fill_between(t, 0, rates, color=colors['rate'], alpha=0.15)
    ax4.set_ylabel('Taxa de disparo\n(Hz)', color=colors['text'], fontsize=11)
    ax4.set_xlabel('Tempo (s)', color=colors['text'], fontsize=11)
    ax4.set_xlim(0, T)
    ax4.tick_params(colors=colors['text'])
    ax4.grid(True, alpha=0.15, color=colors['grid'])
    for spine in ['top', 'right']:
        ax4.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax4.spines[spine].set_color(colors['grid'])

    info_text = (f"Modelo: LIF  |  τ_m = 10ms  |  V_thresh = -45mV  |  "
                 f"Total spikes = {len(spike_times)}  |  Taxa média = {len(spike_times)/T:.0f} Hz")
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=9,
             color=colors['text'], alpha=0.6, style='italic')
    fig.text(0.5, 0.005, 'Parâmetros: Borst & Haag (1996) · de Ruyter van Steveninck (1997)',
             ha='center', fontsize=7, color=colors['text'], alpha=0.4)

    output_path = 'h1_simulation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    print(f"\n✓ Gráfico salvo em {output_path}")

    # Transfer function
    fig2, ax = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    vels = np.linspace(0, 1000, 50)
    analytical_rates = [h1.get_rate_for_velocity(v) for v in vels]
    ax.plot(vels, analytical_rates, color='#00d4ff', linewidth=2.5)
    ax.fill_between(vels, 0, analytical_rates, color='#00d4ff', alpha=0.1)
    ax.set_xlabel('Velocidade angular (°/s)', color='#e0e0e0', fontsize=12)
    ax.set_ylabel('Taxa de disparo (Hz)', color='#e0e0e0', fontsize=12)
    ax.set_title('Transfer Function H1: Velocidade → Spike Rate',
                 color='#e0e0e0', fontsize=14, fontweight='bold')
    ax.tick_params(colors='#e0e0e0')
    ax.grid(True, alpha=0.15, color='#1a1a2e')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#1a1a2e')

    ax.axvspan(0, 30, alpha=0.05, color='#888888', label='Parada/drift')
    ax.axvspan(30, 150, alpha=0.05, color='#00ff88', label='Caminhada')
    ax.axvspan(150, 400, alpha=0.05, color='#ff6b35', label='Vôo normal')
    ax.axvspan(400, 1000, alpha=0.05, color='#ff3366', label='Saccade')
    ax.legend(fontsize=9, facecolor='#0a0a0a', edgecolor='#1a1a2e',
              labelcolor='#e0e0e0', loc='lower right')

    plt.savefig('h1_transfer_function.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    print("✓ Transfer function salva em h1_transfer_function.png")

    # Mostrar se tiver display
    try:
        plt.show()
    except Exception:
        pass
