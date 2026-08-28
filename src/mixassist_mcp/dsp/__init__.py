"""Algoritmos DSP del servidor.

Cada módulo implementa un análisis concreto y no sabe nada de MCP: recibe
audio ya cargado (numpy) y devuelve estructuras de datos planas. Así los
algoritmos se pueden probar sin levantar el servidor, y `server.py` queda
como una capa delgada que solo traduce entre MCP y estas funciones.
"""
