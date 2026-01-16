import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_stack(stack):
    n_modes = stack.shape[0]
    ny, nx = stack.shape[-2:]
    aspect = ny / nx
    width = 12
    height = max(5, width * aspect / 2)  # minimum height
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height))
    plt.subplots_adjust(bottom=0.2, top=0.90, left=0, right=1)
    
    # Intensité et phase
    im1 = ax1.imshow(jnp.abs(stack[0]), cmap='inferno')
    im2 = ax2.imshow(jnp.angle(stack[0]), cmap='twilight')
    im2.set_clim(-jnp.pi, jnp.pi)
    cbar = fig.colorbar(im2, ax=ax2, ticks=[-jnp.pi, 0, jnp.pi])
    cbar.ax.set_yticklabels(['-π', '0', 'π'])
    ax1.axis('off')
    ax2.axis('off')
    
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Mode', 0, n_modes-1, valinit=0, valstep=1)
    
    def update(val):
        idx = int(slider.val)
        data_abs = jnp.abs(stack[idx])
        data_angle = jnp.angle(stack[idx])
        im1.set_data(data_abs)
        im1.set_clim(data_abs.min(), data_abs.max())
        im2.set_data(data_angle)
        fig.canvas.draw_idle()
    
    def on_key(event):
        if event.key == 'right':
            slider.set_val(min(slider.val + 1, n_modes-1))
        elif event.key == 'left':
            slider.set_val(max(slider.val - 1, 0))
    
    slider.on_changed(update)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
