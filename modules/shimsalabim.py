# ShimSalaBim core implementation will go here
# shim_salam_bim_mitm.py



import sys
import importlib
import types
from rich import print as rp
from datetime import datetime
import json
import os
import functools



class MethodMonitor:
    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__

    def __call__(self, *args, **kwargs):
        try:

            rp(f"[ShimSalaBim MITM] Calling {self.name} with args={args}, kwargs={kwargs}")
            result = self.func(*args, **kwargs)
            rp(f"[ShimSalaBim MITM] {self.name} returned {result}")
            return result
        except (ImportError, SystemError):
            # Python is shutting down, just call the original function
            return self.func(*args, **kwargs)
        except Exception as e:
            # For any other errors, still try to call the original function
            try:

                rp(f"[ShimSalaBim MITM] Error in wrapper: {e}")
            except:
                pass
            return self.func(*args, **kwargs)

class ProxyClass:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, attr):
        attr_val = getattr(self._obj, attr)
        if callable(attr_val):
            # Wrap callable attribute to monitor calls
            return MethodMonitor(attr_val, name=f"{self._obj.__class__.__name__}.{attr}")
        return attr_val

class ShimSalaBim:
    def __init__(self, pkg_info_list, classes_to_wrap=None):
        """
        pkg_info_list: list of tuples (package_name, site_packages_path)
        classes_to_wrap: dict of {package_name: [class_names_to_wrap]}
        """
        self._pkg_paths = {}
        self._pkgs = {}
        self._classes_to_wrap = classes_to_wrap or {}
        self.package_names = []  # list of package names

        for pkg_name, path in pkg_info_list:
            self._pkg_paths[pkg_name] = path
            self.package_names.append(pkg_name)

    def __getattr__(self, pkg_name):
        if pkg_name in self._pkgs:
            return self._pkgs[pkg_name]

        path = self._pkg_paths.get(pkg_name)
        if path and path not in sys.path:
            sys.path.append(path)

        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            raise AttributeError(f"Global package '{pkg_name}' not found in path '{path}'")

        # Wrap specified classes in the package if any
        classes_to_wrap = self._classes_to_wrap.get(pkg_name, [])

        if classes_to_wrap:
            for cls_name in classes_to_wrap:
                orig_cls = getattr(pkg, cls_name, None)
                if orig_cls:
                    # Replace with wrapped class proxy
                    wrapped_cls = self._make_proxy_class(orig_cls)
                    setattr(pkg, cls_name, wrapped_cls)

        self._pkgs[pkg_name] = pkg
        return pkg

    def _make_proxy_class(self, orig_cls):
        # Proxy class wrapping original class instances
        class Wrapped(orig_cls):
            def __init__(self, *args, **kwargs):
                try:
                    from rich import print as rp
                    rp(f"[ShimSalaBim MITM] Creating instance of {orig_cls.__name__} with args={args}, kwargs={kwargs}")
                except:
                    pass
                super().__init__(*args, **kwargs)

            def __getattribute__(self, name):
                attr = super().__getattribute__(name)
                if callable(attr) and not name.startswith("__"):
                    return MethodMonitor(attr, name=f"{orig_cls.__name__}.{name}")
                return attr

        Wrapped.__name__ = orig_cls.__name__
        Wrapped.__doc__ = orig_cls.__doc__
        return Wrapped

    def list_loaded_packages(self):
        return list(self._pkgs.keys())

    def cleanup(self):
        """Clean shutdown to avoid destructor issues."""
        try:
            for pkg_name in list(self._pkgs.keys()):
                if hasattr(self._pkgs[pkg_name], '__del__'):
                    try:
                        del self._pkgs[pkg_name]
                    except:
                        pass
            self._pkgs.clear()
        except:
            pass

def init_shims():
    # Example usage
    try:
        
        
        global_packages_folder = '/home/codemonkeyxl/.local/lib/python3.10/site-packages'
        
        global_pkgs = [
            ('llama_cpp', global_packages_folder),
            ('torch', global_packages_folder),
            ('torchvision', global_packages_folder),
            ('langchain', global_packages_folder),
            ('langchain_community', global_packages_folder),
            ('accelerate', global_packages_folder),
            ('safetensors', global_packages_folder),
            ('gguf', global_packages_folder),
        ]


        # Specify which classes you want to monitor usage for
        classes_to_monitor = {
            'llama_cpp': ['Llama'],
            'torch': ['nn.Module'],
            'torchvision': ['transforms.Compose'],
            'langchain_huggingface': ['transformers.pipeline'], 
            'langchain_community': ['embeddings.HuggingFaceEmbeddings'],
            'accelerate': ['AcceleratorState'],
            'safetensors': ['safetensors.torch.SFT', 'safetensors.torch.SFT.from_pretrained'],
            'gguf': ['gguf.GGUF'],
        }

        shim = ShimSalaBim(global_pkgs, classes_to_wrap={})
        
        Llama = shim.llama_cpp.Llama
        torch = shim.torch
        torchvision = shim.torchvision
        langchain_huggingface = shim.langchain_huggingface
        langchain_community = shim.langchain_community
        accelerate = shim.accelerate
        safetensors = shim.safetensors
        gguf = shim.gguf

        if not Llama:
            from llama_cpp import Llama
        
        return Llama


    except Exception as e:
        try:
            from rich import print as rp
            rp(f"[!] Error in main: {e}")
            return None
        except:
            print(f"[!] Error in main: {e}")
            return None
    finally:
        # Clean shutdown
        try:
            if 'shim' in locals():
                shim.cleanup()
        except:
            pass

