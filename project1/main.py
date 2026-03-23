from fastapi import FastAPI, Path, HTTPException
from typing import Optional
from pydantic import BaseModel
import json
from pathlib import Path as FilePath

app = FastAPI()

# File path
DATA_FILE = FilePath("project1/store.json")



# DEFAULT DATA 

default_students = {
    1: {
        "name": "Alice",
        "age": 20,
        "grade": "A",
        "class": "Math"
    }
}



# JSON HANDLING

def load_data():
    if DATA_FILE.exists():
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            if data:  # if file has data
                return {int(k): v for k, v in data.items()}
    return default_students.copy()


def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


# Load data at startup
students = load_data()
save_data(students)  # ensure file exists with default data



#  PYDANTIC MODELS
#request body for create and update student
class Student(BaseModel):
    name: str
    age: int
    grade: str
    class1: str


class UpdateStudent(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    grade: Optional[str] = None
    class1: Optional[str] = None



# ROUTES

@app.get("/")
def home():
    return {"message": "Hello, World!"}


@app.get("/get-students/{student_id}")
def get_student(
    student_id: int = Path(..., description="Student ID", gt=0)
):
    if student_id in students:
        return students[student_id]
    raise HTTPException(status_code=404, detail="Student not found")


@app.get("/get-students-by-name")
def get_student_by_name(name: str):
    for student_id in students:
        if students[student_id]["name"].lower() == name.lower():
            return students[student_id]
    raise HTTPException(status_code=404, detail="Student not found")



#  CREATE

@app.post("/create-student/{student_id}")
def create_student(student_id: int, student: Student):
    if student_id in students:
        raise HTTPException(status_code=400, detail="Student already exists")

    students[student_id] = {
        "name": student.name,
        "age": student.age,
        "grade": student.grade,
        "class": student.class1
    }

    save_data(students)
    return students[student_id]



# UPDATE

@app.put("/update-student/{student_id}")
def update_student(student_id: int, student: UpdateStudent):
    if student_id not in students:
        raise HTTPException(status_code=404, detail="Student not found")

    if student.name is not None:
        students[student_id]["name"] = student.name

    if student.age is not None:
        students[student_id]["age"] = student.age

    if student.grade is not None:
        students[student_id]["grade"] = student.grade

    if student.class1 is not None:
        students[student_id]["class"] = student.class1

    save_data(students)
    return students[student_id]



#  DELETE

@app.delete("/delete-student/{student_id}")
def delete_student(student_id: int):
    if student_id not in students:
        raise HTTPException(status_code=404, detail="Student not found")

    del students[student_id]
    save_data(students)

    return {"message": "Student deleted successfully"}