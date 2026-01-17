import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_stack(stack):
    n_elems = stack.shape[0]
    ny, nx = stack.shape[-2:]
    aspect = ny / nx
    is_complex = jnp.iscomplexobj(stack)
    
    if is_complex:
        width = 14
        height = max(5, width * aspect / 2.5)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height))
        axes = [ax1, ax2]
        transforms = [jnp.abs, jnp.angle]
        cmaps = ['inferno', 'twilight']
        clims = [None, (-jnp.pi, jnp.pi)]
    else:
        width = 6
        height = max(5, width * aspect / 1.1)
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        axes = [ax]
        transforms = [lambda x: x]
        cmaps = ['viridis']
        clims = [None]
    
    plt.subplots_adjust(bottom=0.2, top=0.90, left=0.1, right=0.9)
    
    # Initialize images
    images = []
    for ax, trans, cmap, clim in zip(axes, transforms, cmaps, clims):
        im = ax.imshow(trans(stack[0]), cmap=cmap)
        if clim:
            im.set_clim(*clim)
        if is_complex and trans == jnp.angle:
            cbar = fig.colorbar(im, ax=ax, ticks=[-jnp.pi, 0, jnp.pi])
            cbar.ax.set_yticklabels(['-π', '0', 'π'])
        elif not is_complex:
            fig.colorbar(im, ax=ax)
        ax.axis('off')
        images.append(im)
    
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Index', 1, n_elems, valinit=1, valstep=1)
    
    def update(val):
        idx = int(slider.val - 1)
        for im, trans, clim in zip(images, transforms, clims):
            data = trans(stack[idx])
            im.set_data(data)
            if not clim:  # auto clim for intensity/real values
                im.set_clim(data.min(), data.max())
        fig.canvas.draw_idle()
    
    def on_key(event):
        if event.key == 'right':
            slider.set_val(min(slider.val + 1, n_elems))
        elif event.key == 'left':
            slider.set_val(max(slider.val - 1, 1))
    
    slider.on_changed(update)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    
def visualize_intensity(*stacks):
    """Display total intensity of multiple stacks side by side.
    
    Args:
        *stacks: variable number of (n_modes, ny, nx) arrays or ScalarField
    """
    n_stacks = len(stacks)
    fig, axes = plt.subplots(1, n_stacks, figsize=(4*n_stacks, 4))
    if n_stacks == 1:
        axes = [axes]
    
    for ax, stack in zip(axes, stacks):
        if stack[:].ndim == 2:
            total_intensity = jnp.abs(stack[:])**2
        else:
            total_intensity = jnp.sum(jnp.abs(stack[:])**2, axis=0)
        ax.imshow(total_intensity, cmap='inferno')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
