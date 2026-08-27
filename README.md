# OpenCV

це мій практичний проєкт з OpenCV на Python, який я виконував на основі навчального репозиторію [jasmcaus/opencv-course](https://github.com/jasmcaus/opencv-course).

ідея роботи була ознайомитись з інструментами - запускати код, дивитися, як змінюється зображення, міняти параметри, ловити помилки, виправляти їх і коротко пояснювати, що саме відбувається

Основна частина роботи зібрана в Jupyter Notebook:

```text
opencv_course_analysis.ipynb
```

У ноутбуці збережені результати виконання, графіки, зображення та короткі пояснення до нових функцій і методів
якщо функція вже була описана раніше, повторно я її не розписував

---

## Що є в роботі

У межах курсу я пройшов чотири основні секції.

### Section 1 — Basics

тут зібрані базові речі, з яких починається робота з OpenCV:

- читання зображень;
- робота з відео;
- grayscale;
- Gaussian Blur;
- Canny Edge Detection;
- dilation та erosion;
- resize і crop;
- пошук контурів;
- малювання фігур і тексту;
- thresholding;
- adaptive thresholding;
- translation, rotation і flip.

розділ був потрібен, щоб звикнути до того, що зображення в OpenCV це фактично нампай-масив, а більшість операцій це робота зі значеннями пікселів та їх координатами

### Section 2 — Advanced

- bitwise operations;
- average blur;
- Gaussian blur;
- median blur;
- bilateral filtering;
- робота з BGR, RGB, HSV і LAB;
- Laplacian;
- Sobel;
- гістограми;
- masking;
- split і merge каналів;
- масштабування зображень і відеокадрів.

у цьому розділі я окремо дивився, як параметри функцій впливають на результат
наприклад, як змінюється blur при іншому kernel size або як пороги Canny впливають на кількість знайдених границь

### Section 3 — Faces

у третій секції працював з обличчями:

- Haar Cascade;
- `CascadeClassifier`;
- `detectMultiScale`;
- пошук облич на фотографії;
- підготовка навчальних зображень;
- LBPH Face Recognizer;
- навчання моделі;
- збереження навченої моделі;
- повторне використання моделі для розпізнавання.

цей розділ був особливо корисним, тому що тут уже з'являється не просто обробка картинки, а повноцінний пайплайн, тобто знайти обличчя, потім вирізати ROI, підготувати ознаки, навчити модель, зробити предікт.

### Section 4 — Capstone

останній розділ — невеликий проєкт із класифікації персонажів сімпсонів.

тут я вже зробив повний цикл:

1. автоматично завантажив датасет із Kaggle;
2. проаналізував кількість зображень у класах;
3. вибрав 10 найбільших класів;
4. підготував зображення через OpenCV;
5. перевів їх у grayscale;
6. привів до розміру `80 × 80`;
7. нормалізував пікселі;
8. поділив дані на train і validation;
9. додав аугментації;
10. побудував CNN у TensorFlow/Keras;
11. навчив модель;
12. перевірив accuracy та loss;
13. зробив прогноз для окремого тестового зображення;
14. зберіг модель у файл.

---

## Структура проєкту

```text
opencv-course/
│
├── Section #1 - Basics/
├── Section #2 - Advanced/
├── Section #3 - Faces/
├── Section #4 - Capstone/
│
├── Resources/
│   ├── Photos/
│   ├── Videos/
│   └── Faces/
│
├── data/
│
├── opencv_course_analysis.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

Папка `data/` створюється під час роботи Section 4
---

## Середовище

для роботи я використовував окреме Conda-середовище

```bash
conda create -n opencv-practice python=3.12 -y
conda activate opencv-practice
```

після цього встановлюються залежності

```bash
python -m pip install --upgrade pip
python -m pip install opencv-contrib-python numpy matplotlib jupyter ipykernel tensorflow scikit-learn kagglehub
```

`opencv-contrib-python` використовую тому, що він містить не тільки основний `cv2`, а й додаткові модулі, зокрема `cv.face`, який потрібен для LBPH Face Recognizer.

Щоб середовище було доступне в Jupyter

```bash
python -m ipykernel install --user --name opencv-practice --display-name "Python (opencv-practice)"
```

Після цього ноутбук запускається командою

```bash
jupyter notebook
```

і в ноутбуці обрав kernel

```text
Python (opencv-practice)
```

---

після клонування репозиторію:

```bash
git clone https://github.com/vladdyl20/opencv-course.git
```

потрібно перейти в папку

```bash
cd opencv-course
```

переключитися на робочу гілку

```bash
git switch feature/opencv-practice
```

активувати середовище

```bash
conda activate opencv-practice
```

і запустити ноутбук

```bash
jupyter notebook
```

після цього відкривається

```text
opencv_course_analysis.ipynb
```

---

## датасет для Capstone

датасет у Section 4 завантажується прямо з ноутбук через `kagglehub`.

```python
import kagglehub

dataset_path = kagglehub.dataset_download(
    "alexattia/the-simpsons-characters-dataset",
    output_dir="data"
)
```

після завантаження код сам знаходить

```text
simpsons_dataset
kaggle_simpson_testset
```

тому вручну завантажувати архів і додавати його в репозиторій не потрібно було

---

## Що вийшло в Capstone

у датасеті було знайдено 43 класи

для першого експерименту я використав 10 найбільших класів і по 500 зображень на кожен клас

У результаті вийшло

```text
5000 зображень
```

після preprocessing

```text
X shape: (5000, 80, 80, 1)
y shape: (5000,)
```

поділ даних

```text
Training:   4000
Validation: 1000
```

тобто для кожного з 10 класів

```text
400 train
100 validation
```

після 10 епох модель показала

```text
Validation Accuracy: 25.90%
Validation Loss:     2.1233
```

сам pipeline працює, тобто датасет завантажується, preprocessing виконується, навчається модель і робить прогноз, в кінці зберігається


## в ноутбуці - 

для нових функцій я додавав короткі Markdown-коментарі

там пояснюється
-що робить функція
-що приблизно відбувається на рівні пікселів
-які параметри найбільше впливають на результат
-які є альтернативи

повторні вже описані функцій я не дублював, щоб ноутбук не був перевантажений однаковим текстом

---

## альтернативи OpenCV

під час роботи я також дивився, чим можна замінити окремі інструменти OpenCV

Наприклад:

- Pillow — просте читання та обробка зображень
- scikit-image — класичні алгоритми image processing
- Albumentations — аугментація
- TorchVision — preprocessing і Deep Learning у PyTorch
- TensorFlow / Keras — нейронні мережі та inference

---

## гітхаб

робота виконується в окремій гілці:

```text
feature/opencv-practice
```

зміни фіксувалися невеликими комітами


після завершення окремого етапу:

```bash
git add .
git commit -m "feat: describe completed section"
git push
```

## короткий висновок

цей курс дав хороший базовий маршрут від простих операцій OpenCV до невеликого проєкту.

на початку були звичайні речі у вигляді прочитати зображення, змінити розмір, знайти границі, намалювати контури
далі з'явилися masking, histograms, color spaces, face detection, face recognition
---

