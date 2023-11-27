class Persona():

    def __init__(self, nombre_completo, dni, fecha_nac, sexo, residencia):
        self.nombre_completo = nombre_completo
        self.dni = dni
        self.fecha_nac = fecha_nac
        self.sexo = sexo
        self.residencia = residencia
        return
    


class Empleado(Persona):


    def __init__(self, nombre_completo,dni,fecha_nac,sexo,residencia,rol,rango,antiguedad):
        super().__init__(nombre_completo,dni,fecha_nac,sexo,residencia)
        self.rol = rol
        self.rango = rango
        self.antiguedad = antiguedad
        return
    

    def __str__(self):
        return  f"""Nombre Completo: {self.nombre_completo},
DNI: {self.dni}
Fecha de Nacimiento: {self.fecha_nac},
Sexo: {self.sexo}
Residencia: {self.residencia}
Rol: {self.rol}
Rango: {self.rango}
Antiguedad: {self.antiguedad} años
"""




