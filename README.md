![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)
![Vulkan](https://img.shields.io/badge/Vulkan-1.3-AC162C?style=for-the-badge&logo=vulkan&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)

# rviv 🖼️⚡

**rviv** is a blazing-fast, minimalist image viewer and wallpaper setter for Linux, powered by **Vulkan 1.3** and written in **Rust**.

Designed for performance and keyboard-centric workflows, it features multi-threaded image decoding, GPU-accelerated rendering, and smart texture caching. The release build is heavily optimized with Link-Time Optimization (LTO) and single-codegen-unit compilation to ensure instant navigation, even with massive image collections.

## ✨ Features

- **Vulkan 1.3 Powered:** Uses dynamic rendering for highly optimized, low-overhead GPU image drawing.
- **Heavily Optimized Binary:** Release builds use `lto = true`, `codegen-units = 1`, and `strip = true` for maximum performance and minimal footprint.
- **Non-Blocking UI:** Images are decoded in a background thread, keeping the interface 100% responsive.
- **Smart Caching:** Implements an LRU texture cache (up to 10 images) to avoid redundant GPU uploads when navigating back and forth.
- **Vim-like Keybindings:** Navigate, pan, and zoom your images without ever leaving the home row.
- **Built-in Wallpaper Setter:** Set images as your X11 desktop background directly from the CLI (fully compatible with compositors like `picom` via 32-bit pixmaps).
- **Slideshow Mode:** Automatically cycle through images at a custom interval.
- **Broad Format Support:** JPEG, PNG, GIF, BMP, WebP, TIFF, and ICO.
- **Cross-Display Server:** The viewer runs natively on both **X11** and **Wayland** (via `winit`).

## 📦 Installation

### System Dependencies

Since `rviv` uses `winit` (X11/Wayland) and `x11rb`, you might need standard graphics development libraries installed on your system:

- **Arch Linux:** `sudo pacman -S libx11 libxkbcommon wayland`
- **Debian/Ubuntu:** `sudo apt install libx11-dev libxkbcommon-x11-dev libwayland-dev`

### Build from source

The pre-compiled SPIR-V shaders are included in the repository, so building is straightforward:

```bash
git clone https://github.com/azytar/rviv.git
cd rviv

# Build the highly-optimized release binary
cargo build --release
```

The binary will be located at `target/release/rviv`. You can move it to your path:

```bash
cp target/release/rviv ~/.local/bin/
```

## 🚀 Usage

`rviv` can be used as a standalone image viewer or as a CLI wallpaper setter.

### View Images

```bash
# Open a single image
rviv image.jpg

# Open a directory (non-recursive)
rviv ~/Pictures

# Open a directory recursively and sort alphabetically
rviv -r --sort ~/Wallpapers

# Start in fullscreen with a 5-second slideshow
rviv -F -S 5.0 ~/Pictures
```

### Set Wallpaper (X11)

Use the `--bg` flag to set an image as your desktop wallpaper and exit. This sets the `_XROOTPMAP_ID` and `ESETROOT_PMAP_ID` properties, making it compatible with transparent terminals and compositors.

```bash
rviv --bg ~/Wallpapers/landscape.png
```

> Note: Wayland wallpaper setting is not supported; use tools like `swww` or `hyprpaper` for Wayland.

## ⌨️ Keybindings

| Action | Keys |
|---------|------|
| **Next Image** | `l` or `Shift + Right` |
| **Previous Image** | `h` or `Shift + Left` |
| **Pan (Up/Down)** | `k` / `j` or `Up` / `Down` |
| **Pan (Left/Right)** | `Left` / `Right` |
| **Zoom In** | `+` or `Mouse Wheel Up` |
| **Zoom Out** | `-` or `Mouse Wheel Down` |
| **Reset Zoom/Pan** | `0` |
| **Toggle Fullscreen** | `f` |
| **Quit** | `q` or `Escape` |

## 🛠️ Technical Details & Stack

Under the hood, `rviv` relies on a modern, low-level Rust stack:

- **`ash` (0.38) & `ash-window`:** Raw Vulkan 1.3 bindings for zero-cost GPU abstraction.
- **`winit` (0.29):** Cross-platform window creation and event loop handling.
- **`x11rb`:** Pure Rust X11 client for the wallpaper setter functionality.
- **`image`:** Multi-format image decoding.

### Rendering Architecture

- Uses a simple vertex/fragment shader pipeline with **push constants** to handle dynamic image scaling and offsetting without needing to recreate vertex buffers.
- Dynamically resizes the staging buffer if an image exceeds the current capacity, ensuring smooth uploads of high-resolution textures.
- Uses `mpsc` channels to communicate between the main rendering thread and the background image decoding worker thread.

### Modifying Shaders

If you want to tweak the shaders, the GLSL source files are located in `shaders/` (`vert.vert` and `frag.frag`). To recompile them into SPIR-V (which is embedded via `include_bytes!` in `main.rs`), use `glslc`:

```bash
glslc shaders/vert.vert -o shaders/vert.spv
glslc shaders/frag.frag -o shaders/frag.spv
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
