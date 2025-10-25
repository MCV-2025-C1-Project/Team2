# Imports and server utilities
import os
import imageio
import numpy as np
from scipy import ndimage
from flask import Flask, request, jsonify
import threading
import cv2

# Note: if Flask is not installed: pip install flask


# Build image list and mask map
image_folder = "./Data/Week2/qsd2_w2/"  # Update this path if needed
all_files = os.listdir(image_folder)
# Only consider JPG/JPEG files as primary images to visualize
image_files = sorted([f for f in all_files if f.lower().endswith(('.jpg', '.jpeg'))])
# Map of PNG mask files by basename (if available)
mask_files = { os.path.splitext(f)[0]: f for f in all_files if f.lower().endswith('.png') }
print('Found', len(image_files), 'images and', len(mask_files), 'mask files (png)')


# Morphological image-level function (accepts full RGB image).
# p3 and p4 are additional placeholder parameters and are not used.
def morph_image(image, close_size=9, open_size=5, p1=0.0, p2=1.0, p3=None, p4=None):
    
    img = image.astype(np.float32)
    
    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]
    # compute grayscale (luminance) from original RGB if available
    
    gray = L.copy()

    def proc_channel(ch):
        c = ch.astype(float)
        #We apply the median filter to reduce noise
        return c

    zl = proc_channel(L)
    za = proc_channel(A)
    zb = proc_channel(B)
    zgray = proc_channel(gray)

    return {'zl': zl, 'za': za, 'zb': zb, 'zgray': zgray}


# Prepare data for a given image filename: returns xs, ys, zl, za, zb, zgray, zgrad and a downsampled mask (flat lists)
def prepare_data(image_filename, downsample=6, close_size=9, open_size=5, p1=0.0, p2=1.0, p3=None, p4=None):
    path = os.path.join(image_folder, image_filename)
    im = imageio.imread(path)
    if im.ndim == 2:
        # convert grayscale to RGB by duplicating channels
        im = np.stack([im, im, im], axis=-1)
    # call the image-level morph function
    processed = morph_image(im, close_size=close_size, open_size=open_size, p1=p1, p2=p2, p3=p3, p4=p4)
    zl = processed['zl']
    za = processed['za']
    zb = processed['zb']
    zgray = processed['zgray']
    H, W = zl.shape

    # load associated PNG mask if it exists (same basename)
    mask_path = os.path.splitext(path)[0] + '.png'
    if os.path.exists(mask_path):
        m = imageio.imread(mask_path)
        if m.ndim == 3:
            m = m[:, :, 0]
        # binarize: assume white > 127
        mask_full = (m > 127).astype(np.uint8)
        # if mask size differs from image size, rescale using nearest-neighbor
        if mask_full.shape != (H, W):
            zoom_y = H / mask_full.shape[0]
            zoom_x = W / mask_full.shape[1]
            mask_full = ndimage.zoom(mask_full.astype(float), (zoom_y, zoom_x), order=0).astype(np.uint8)
    else:
        # if no mask exists, assume all white (1)
        mask_full = np.ones((H, W), dtype=np.uint8)

    ys, xs = np.mgrid[0:H:downsample, 0:W:downsample]
    zl_s = zl[0:H:downsample, 0:W:downsample].astype(float)
    za_s = za[0:H:downsample, 0:W:downsample].astype(float)
    zb_s = zb[0:H:downsample, 0:W:downsample].astype(float)
    zgray_s = zgray[0:H:downsample, 0:W:downsample].astype(float)
    mask_s = mask_full[0:H:downsample, 0:W:downsample].astype(int)

    # compute morphological gradient of the full-resolution grayscale using p3,p4 as structuring element sizes
    # default to a small 3x3 structure if p3/p4 are not provided or invalid
    try:
        p3_i = int(p3) if p3 is not None else 3
    except Exception:
        p3_i = 3
    try:
        p4_i = int(p4) if p4 is not None else 3
    except Exception:
        p4_i = 3
    try:
        struct = np.ones((max(1, p3_i), max(1, p4_i)))
        zgrad = ndimage.morphological_gradient(zgray, structure=np.ones((5, 5)))
    except Exception:
        # fallback: zero gradient
        zgrad = np.zeros_like(zgray)

    zgrad_s = zgrad[0:H:downsample, 0:W:downsample].astype(float)

    xs_flat = xs.ravel().tolist()
    ys_flat = ys.ravel().tolist()
    zl_flat = zl_s.ravel().tolist()
    za_flat = za_s.ravel().tolist()
    zb_flat = zb_s.ravel().tolist()
    zgray_flat = zgray_s.ravel().tolist()
    zgrad_flat = zgrad_s.ravel().tolist()
    mask_flat = mask_s.ravel().tolist()

    return {
        'xs': xs_flat,
        'ys': ys_flat,
        'zl': zl_flat,
        'za': za_flat,
        'zb': zb_flat,
        'zgray': zgray_flat,
        'zgrad': zgrad_flat,
        'mask': mask_flat,
        'width': W, 'height': H,
    }

# Flask app serving the UI and data endpoints
app = Flask(__name__)

@app.route('/images')
def list_images():
    return jsonify(image_files)

@app.route('/data')
def get_data():
    image = request.args.get('image')
    downsample = int(request.args.get('downsample', 6))
    # parse integer sizes and float p1,p2,p3,p4 (p3/p4 are optional placeholders used for gradient structuring element)
    close_size = int(request.args.get('close_size', 9))
    open_size = int(request.args.get('open_size', 5))
    try:
        p1 = float(request.args.get('p1', 0.0))
    except Exception:
        p1 = 0.0
    try:
        p2 = float(request.args.get('p2', 1.0))
    except Exception:
        p2 = 1.0
    # optional placeholders (p3/p4 used as integers for morphological gradient structure)
    try:
        p3 = int(request.args.get('p3')) if request.args.get('p3') else 3
    except Exception:
        p3 = 3
    try:
        p4 = int(request.args.get('p4')) if request.args.get('p4') else 3
    except Exception:
        p4 = 3

    if not image or image not in image_files:
        return jsonify({'error': 'image not found'}), 400
    data = prepare_data(image, downsample=downsample, close_size=close_size, open_size=open_size, p1=p1, p2=p2, p3=p3, p4=p4)
    return jsonify(data)

@app.route('/')
def index():
    # HTML + JS: dropdown + inputs for parameters + five Plotly divs (including morphological gradient)
    html = '''
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
  <style>
    body{font-family:Arial;margin:10px}
    #controls{display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap}
    #row{display:flex;gap:10px}
    .plotbox{flex:1;height:420px;border:1px solid #eee;padding:4px;background:#fafafa;min-width:180px}
  </style>
</head>
<body>
  <h3>Select image</h3>
  <div id='controls'>
    <select id='sel' style='min-width:300px'></select>
    <label> Downsample: <input id='ds' type='number' value='6' min='1' max='100' style='width:60px'></label>
    <label> Close: <input id='close' type='number' value='9' min='1' max='101' style='width:60px'></label>
    <label> Open: <input id='open' type='number' value='5' min='1' max='101' style='width:60px'></label>
    <label> p1 (sigma): <input id='p1' type='number' value='0' step='0.1' style='width:60px'></label>
    <label> p2 (scale): <input id='p2' type='number' value='1' step='0.1' style='width:60px'></label>
    <label> p3: <input id='p3' type='number' value='3' style='width:60px'></label>
    <label> p4: <input id='p4' type='number' value='3' style='width:60px'></label>
    <button id='btn'>Load</button>
    <span id='status' style='margin-left:8px;color:#555'></span>
  </div>
  <div id='row'>
    <div id='plotA' class='plotbox'></div>
    <div id='plotB' class='plotbox'></div>
    <div id='plotL' class='plotbox'></div>
    <div id='plotGray' class='plotbox'></div>
    <div id='plotGrad' class='plotbox'></div>
  </div>
  <script>
    // debounce helper
    function debounce(fn, wait){ let t=null; return function(...args){ clearTimeout(t); t=setTimeout(()=>fn.apply(this,args), wait); }; }

    async function fill(){
      const res = await fetch('/images');
      const imgs = await res.json();
      const sel = document.getElementById('sel'); sel.innerHTML='';
      imgs.forEach(i=>{ const o=document.createElement('option'); o.value=i; o.text=i; sel.appendChild(o); });
      if(imgs.length>0) sel.selectedIndex=0;
    }

    // helper: build per-point color using mask and small value-based lightness variation
    function buildColors(values, mask, insideHue=210, outsideHue=30){
      const eps = 1e-12;
      let vmin = Infinity, vmax = -Infinity;
      for(let v of values){ if(!isNaN(v)){ if(v<vmin) vmin=v; if(v>vmax) vmax=v; }}
      const denom = (vmax - vmin) + eps;
      const colors = new Array(values.length);
      for(let i=0;i<values.length;i++){
        const v = values[i];
        const t = (v - vmin) / denom; // 0..1
        // map t to lightness between 35% and 75%
        const light = Math.round(35 + t*40);
        const hue = mask[i] ? insideHue : outsideHue;
        colors[i] = `hsl(${hue},90%,${light}%)`;
      }
      return colors;
    }

    async function load(){
      const sel = document.getElementById('sel'); const img = sel.value; if(!img) return;
      const btn = document.getElementById('btn'); const status = document.getElementById('status');
      btn.disabled=true; status.textContent='Loading...';
      try{
        const ds = document.getElementById('ds').value || 6;
        const close = document.getElementById('close').value || 9;
        const open = document.getElementById('open').value || 5;
        const p1 = document.getElementById('p1').value || 0;
        const p2 = document.getElementById('p2').value || 1;
        const p3 = document.getElementById('p3').value || 3;
        const p4 = document.getElementById('p4').value || 3;
        const params = new URLSearchParams({image: img, downsample: ds, close_size: close, open_size: open, p1: p1, p2: p2, p3: p3, p4: p4});
        const res = await fetch('/data?'+params.toString());
        const data = await res.json();
        if(data.error){ alert(data.error); status.textContent='Error'; return; }

        const xs = data.xs; const ys = data.ys;
        const za = data.za; const zb = data.zb; const zl = data.zl; const zgray = data.zgray; const zgrad = data.zgrad; const mask = data.mask;

        // build colored markers with slight value-dependent variation per plot
        const colorsA = buildColors(za, mask, 220, 25);
        const colorsB = buildColors(zb, mask, 200, 15);
        const colorsL = buildColors(zl, mask, 150, 35);
        const colorsG = buildColors(zgray, mask, 260, 10);
        const colorsGrad = buildColors(zgrad, mask, 0, 200);

        const traceA = { x: xs, y: ys, z: za, mode:'markers', marker:{size:2, color:colorsA, opacity:0.95}, type:'scatter3d' };
        const traceB = { x: xs, y: ys, z: zb, mode:'markers', marker:{size:2, color:colorsB, opacity:0.95}, type:'scatter3d' };
        const traceL = { x: xs, y: ys, z: zl, mode:'markers', marker:{size:2, color:colorsL, opacity:0.95}, type:'scatter3d' };
        const traceG = { x: xs, y: ys, z: zgray, mode:'markers', marker:{size:2, color:colorsG, opacity:0.95}, type:'scatter3d' };
        const traceGrad = { x: xs, y: ys, z: zgrad, mode:'markers', marker:{size:2, color:colorsGrad, opacity:0.95}, type:'scatter3d' };

        const layoutCommon = (title)=>({title:title,margin:{l:0,r:0,t:30,b:0},scene:{xaxis:{title:'x'},yaxis:{title:'y'},zaxis:{title:'z'}}});
        Plotly.react('plotA',[traceA], layoutCommon(img+' - channel a'));
        Plotly.react('plotB',[traceB], layoutCommon(img+' - channel b'));
        Plotly.react('plotL',[traceL], layoutCommon(img+' - L channel'));
        Plotly.react('plotGray',[traceG], layoutCommon(img+' - grayscale'));
        Plotly.react('plotGrad',[traceGrad], layoutCommon(img+' - morphological gradient'));

        status.textContent='Ready';
      }catch(e){ console.error(e); alert('Error loading data'); status.textContent='Error'; }
      finally{ btn.disabled=false; }
    }

    const debouncedLoad = debounce(load, 300);
    document.getElementById('btn').addEventListener('click', load);
    // trigger load on change for immediate feedback (include p3,p4)
    ['sel','ds','close','open','p1','p2','p3','p4'].forEach(id=>{ const el=document.getElementById(id); if(el) el.addEventListener('change', debouncedLoad); });
    (async ()=>{ await fill(); if(document.getElementById('sel').value) debouncedLoad(); })();
  </script>
</body>
</html>
'''
    return html

# Start the server in a background thread so the notebook is not blocked
def run_server():
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

thread = threading.Thread(target=run_server, daemon=True)
thread.start()
print('Flask server started at http://127.0.0.1:5000 - open that URL in your browser')

try:
    while True:
        pass
except KeyboardInterrupt:
    print("Server stopped.")

