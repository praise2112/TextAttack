import platform
import socket
import psutil
import logging
from subprocess import Popen, PIPE
from xml.etree.ElementTree import fromstring


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def is_nvidia_compatible(*args, **kwargs):
    from shutil import which

    if which("nvidia-smi") is None:
        return False

    # make sure that nvidia-smi doesn't just return no devices
    p = Popen(["nvidia-smi"], stdout=PIPE)
    stdout, stderror = p.communicate()
    output = stdout.decode("UTF-8")
    if "no devices" in output.lower():
        return False

    return True


def get_gpu_info(*args, **kwargs):
    p = Popen(["nvidia-smi", "-q", "-x"], stdout=PIPE)
    outs, errors = p.communicate()
    xml = fromstring(outs)
    datas = []
    driver_version = xml.findall("driver_version")[0].text
    cuda_version = xml.findall("cuda_version")[0].text

    for gpu in xml.getiterator("gpu"):
        name = list(gpu.getiterator("product_name"))[0].text
        memory_usage = gpu.findall("fb_memory_usage")[0]
        total_memory = memory_usage.findall("total")[0].text

        gpu_data = {"name": name, "total_memory": total_memory, "driver_version": driver_version, "cuda_version": cuda_version}

        datas.append(gpu_data)
    return datas


def get_system_info():
    try:
        svmem = psutil.virtual_memory()
        info = {
            'hostname': socket.gethostname(),
            "platform": {
                'type': platform.system(),
                'platform release': platform.release(),
                'platform version': platform.version(),
            },
            'architecture': platform.machine(),
            'processor': {
                "type": platform.processor(),
                "Physical cores": psutil.cpu_count(logical=False),
                "Total cores": psutil.cpu_count(logical=True),
            },
            "memory": {
                "total":  get_size(svmem.total),
                "used": get_size(svmem.used),
                "percentage": svmem.percent
            },
            'ram': f"{str(round(psutil.virtual_memory().total / (1024.0 ** 3)))} GB"
        }
        if is_nvidia_compatible():
            info["gpu"] = get_gpu_info()
        return info
    except Exception as e:
        logging.exception(e)
        return {}

