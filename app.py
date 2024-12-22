import gradio as gr


from gradio import Interface, Button, Blocks, TabbedInterface
# Определяем базу данных и модель
from main_win import iface
from checkup_win import iface2
from analytics_win import iface3



# Объединяем интерфейс и кнопку
iface = TabbedInterface([iface, iface2, iface3], ['Загрузка','Проверка', 'Cтатистика'])

# Запускаем основной интерфейс
iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
