# Importação das bibliotecas necessárias
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pygame

# Variáveis globais para cache do matplotlib
_cached_fig = None
_cached_ax = None
_cached_canvas = None
_last_data_len = 0

def draw_plot(screen: pygame.Surface, x: list, y: list, x_label: str = 'Generation', 
              y_label: str = 'Fitness', use_cache: bool = True) -> None:
    """
    Desenha um gráfico com Pygame.

    Parâmetros:
    - screen (pygame.Surface): A superfície do gráfico do Pygame a ser desenhada.
    - x (list): Os valores do eixo X.
    - y (list): Os valores do eixo Y.
    - x_label (str): Rótulo do eixo X ('Generation').
    - y_label (str): Rótulo do eixo Y ('Fitness').
    - use_cache (bool): Se deve usar a figura em cache para melhor desempenho (padrão é "True").
    """
    global _cached_fig, _cached_ax, _cached_canvas, _last_data_len
    
    # Obter dimensões da tela Pygame
    screen_width, screen_height = screen.get_size()
    fig_width = screen_width / 100  # Converter pixels para polegadas (assumindo 100 DPI)
    fig_height = screen_height / 100
    
    # Se usar cache e já tiver figura criada, apenas atualiza os dados
    if use_cache and _cached_fig is not None and len(y) > 1:
        # Verificar se o tamanho da janela mudou
        current_size = _cached_fig.get_size_inches()
        if abs(current_size[0] - fig_width) > 0.1 or abs(current_size[1] - fig_height) > 0.1:
            # Tamanho mudou, recriar figura
            plt.close(_cached_fig)
            _cached_fig = None
            _cached_ax = None
            _cached_canvas = None
        else:
            # Limpar apenas os dados do plot, mantendo a estrutura
            _cached_ax.clear()
            _cached_ax.plot(x, y, 'b-', linewidth=2)
            _cached_ax.set_ylabel(y_label)
            _cached_ax.set_xlabel(x_label)
            _cached_ax.grid(True, alpha=0.3)
            
            # Redesenhar apenas se houve mudança significativa nos dados
            if len(y) != _last_data_len:
                _cached_canvas.draw()
                _last_data_len = len(y)
            else:
                # Atualização mais leve
                _cached_fig.canvas.draw_idle()
    
    # Criar nova figura se necessário
    if _cached_fig is None or not use_cache:
        _cached_fig, _cached_ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        _cached_canvas = FigureCanvasAgg(_cached_fig)
        
        _cached_ax.plot(x, y, 'b-', linewidth=2)
        _cached_ax.set_ylabel(y_label)
        _cached_ax.set_xlabel(x_label)
        _cached_ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        _cached_canvas.draw()
        _last_data_len = len(y)
    
    # Converter para superfície Pygame
    renderer = _cached_canvas.get_renderer()
    size = _cached_canvas.get_width_height()
    raw_data = renderer.tostring_argb()
    surf = pygame.image.fromstring(raw_data, size, "ARGB")
    
    # Preencher a tela completamente
    screen.fill((255, 255, 255))
    screen.blit(surf, (0, 0))