import sys
import os
import importlib.util as _ilu

_here = os.path.dirname(os.path.abspath(__file__))

# Các sub-package của ComfyUI (utils, app, ...) được import theo kiểu flat,
# ví dụ: `from utils.install_util import ...` thay vì `from ComfyUI.utils...`.
# Trên Colab/Kaggle có thể tồn tại package/module trùng tên (ví dụ PyPI 'utils')
# đã được cache vào sys.modules → gây lỗi "utils is not a package".
# Giải pháp: xóa mọi entry xung đột, thêm _here vào sys.path[0], VÀ nạp
# trực tiếp 'utils' từ đường dẫn tuyệt đối thay vì dùng sys.path search.

_subpkg_names = ('utils', 'app', 'api_server', 'middleware', 'comfy', 'comfy_api',
                 'comfy_execution', 'comfy_extras', 'comfy_config')
for _n in _subpkg_names:
    sys.modules.pop(_n, None)

# Đảm bảo _here luôn ở đầu sys.path (fallback cho các import không được đăng ký bên dưới).
try:
    sys.path.remove(_here)
except ValueError:
    pass
sys.path.insert(0, _here)

# Nạp trực tiếp 'utils' package từ đường dẫn tuyệt đối, hoàn toàn bỏ qua
# mọi sys.path conflict. Điều này đảm bảo `from utils.install_util import ...`
# luôn trỏ đúng vào ComfyUI/utils/, kể cả khi môi trường có utils.py hoặc
# utils/ khác được cài trước.
def _load_subpkg_as_toplevel(name: str) -> None:
    pkg_dir = os.path.join(_here, name)
    init_file = os.path.join(pkg_dir, '__init__.py')
    if not os.path.isfile(init_file):
        return
    spec = _ilu.spec_from_file_location(name, init_file,
                                         submodule_search_locations=[pkg_dir])
    if spec is None:
        return
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod                          # alias top-level
    sys.modules.setdefault(f'ComfyUI.{name}', mod)  # alias ComfyUI.<name>
    spec.loader.exec_module(mod)

_load_subpkg_as_toplevel('utils')
del _ilu, _load_subpkg_as_toplevel
