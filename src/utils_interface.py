import customtkinter as ctk

def center_window(window: ctk.CTk, width: int, height: int) -> None:
    """ Центирование главного окна"""
    # Обновляем геометрию окна
    window.update_idletasks()
    
    screen_width: int = window.winfo_screenwidth()
    screen_height: int = window.winfo_screenheight()
    
    x: int = (screen_width - width) // 2
    y: int = (screen_height - height) // 2
    
    window.geometry(f"{width}x{height}+{x}+{y}")