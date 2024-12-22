DATABASE_URL = "postgresql://root:root@db_postgres:5432/image_db"
from sqlalchemy import create_engine, Column, LargeBinary, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker




Base = declarative_base()

class Defect(Base):
    __tablename__ = 'defects'
    id = Column(Integer, primary_key=True)
    
    image_data = Column(LargeBinary)
    bbox = Column(String)
    class_id = Column(Integer)

# Функция для подключения к базе данных
def connect_db():
    engine = create_engine(DATABASE_URL + "?client_encoding=utf8")
    Base.metadata.create_all(engine)  # Создает таблицы, если они не существуют
    return sessionmaker(bind=engine)()


def paste_to_db(data:Defect):

    session = connect_db()
    print(data)
    session.add(data)
    session.commit()
    session.close()