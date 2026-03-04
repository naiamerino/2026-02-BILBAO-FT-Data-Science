def area_cuadrado (lado):
    area = lado * lado
    return area

def area_triangulo (base, altura):
    area = base*altura/2
    return area

def area_circulo (radio):
    import math
    area = (math.pi*radio)**2
    return area