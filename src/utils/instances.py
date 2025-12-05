import socket


def get_instance_id(base_port=5000, max_instances=4):
    for i in range(max_instances):
        port = base_port + i
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('localhost', port))
            return i + 1, s
        except OSError:
            continue

    raise RuntimeError("Nie udało się przypisać instancji - brak wolnych portów!")