import tkinter as tk
from functools import partial
from tkinter import ttk
from tkinter import filedialog
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


#not use
def color_then_xy_clustering(image, k=5, seed=None):
    """色空間→xy空間クラスタリング"""
    return xy_clustering(color_clustering(image, k, seed))

def color_clustering(image, k=5, seed=None):
    """色空間クラスタリング"""
    # TODO: 今は(224,224)が前提になっているので、任意サイズに対応する
    #色の変更
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).reshape((224**2, 3))
    #sklearnのKMeans法によるクラスタリング
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(lab)
    return kmeans.labels_.reshape((224, 224))
#not use
def xy_clustering(clusters):
    """xy空間クラスタリング"""
    # TODO: 今は(224,224)が前提になっているので、任意サイズに対応する
    M = np.array([[x, y] for x in range(224) for y in range(224)])
    R = np.zeros((224**2,), dtype=int)
    labels = clusters.flatten()
    last_label = 1
    for label in np.unique(labels):
        X = M[labels==label]
        dbscan = DBSCAN(eps=1, min_samples=2).fit(X)
        R[labels==label] = dbscan.labels_ + last_label
        last_label += len(np.unique(dbscan.labels_))
    print(last_label)
    return R.reshape((224, 224))

class Application(tk.Frame):
    STATUS_UNCOMPLETED = 0
    STATUS_DISH_COMPLETED = 1
    STATUS_MASK_COMPLETED = 2

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.version = '0.0.1'
        self.title = f'アノテーションツール ver.{self.version}'
        self.index = 0
        self.image_paths = []
        # TODO: 白、灰、黒、青のどれが良いか？
        self.default_color = [128, 128, 128]
        self.init_input_frame()
        self.init_main_frame()
        self.init_dish_frame()
        self.init_mask_frame()
        self.master.title(self.title)
        self.master.geometry('700x80')
        self.input_frame.tkraise()

    @staticmethod
    def open_directory(entry):
        """ディレクトリを開く"""
        dir_name = filedialog.askdirectory()
        if dir_name:
            entry.delete(0, tk.END)
            entry.insert(tk.END, dir_name)

    def save_dish(self):
        """皿マスクの保存"""
        name = self.image_paths[self.index].stem
        path = self.output_dish_path / f'{name}.png'
        cv2.imwrite(str(path), self.dish)

    def load_dish(self, shape):
        """皿マスクの読み込み"""
        name = self.image_paths[self.index].stem
        path = self.output_dish_path / f'{name}.png'
        if path.exists():
            # 皿マスクが存在する場合は読み込み
            image = cv2.imread(str(path), 0)
            _, self.dish = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        else:
            # 皿マスクが存在しない場合、全領域Trueのマスクを作成
            self.dish = np.ones((shape[0], shape[1]), dtype=np.uint8)

    def save_mask(self):
        """マスクの保存"""
        pass

    def load_mask(self):
        """マスクの読み込み"""
        pass

    def count_up(self):
        """インデックスのカウントアップ"""
        self.index = min(len(self.statuses) - 1, self.index + 1)

    def count_down(self):
        """インデックスのカウントダウン"""
        self.index = max(0, self.index - 1)

    def init_input_frame(self):
        """入力フレームの初期化"""
        # 入力フレーム
        self.input_frame = ttk.Frame(self.master)
        self.input_frame.grid(row=0, column=0, sticky=tk.NSEW)
        # 入力欄
        self.input_label = ttk.Label(self.input_frame, text='入力ディレクトリ') 
        self.input_label.grid(row=0, column=0)
        self.input_entry = ttk.Entry(self.input_frame, width=80)
        self.input_entry.grid(row=0, column=1, sticky=tk.EW)
        self.input_entry.insert(tk.END, 'C:/Users/1001000565/python/input')  # TODO: 検証用。後で削除。
        self.input_button = ttk.Button(self.input_frame, text='参照', command=partial(self.open_directory, self.input_entry))
        self.input_button.grid(row=0, column=2)
        # 出力欄
        self.output_label = ttk.Label(self.input_frame, text='出力ディレクトリ') 
        self.output_label.grid(row=1, column=0)
        self.output_entry = ttk.Entry(self.input_frame, width=80)
        self.output_entry.grid(row=1, column=1, sticky=tk.EW)
        self.output_entry.insert(tk.END, 'C:/Users/1001000565/python/output')  # TODO: 検証用。後で削除。
        self.output_button = ttk.Button(self.input_frame, text='参照', command=partial(self.open_directory, self.output_entry))
        self.output_button.grid(row=1, column=2)
        # 開始ボタン
        self.start_button = ttk.Button(self.input_frame, text='開始', command=self.from_init_to_dish)
        self.start_button.grid(row=2, column=0, columnspan=3)

    def init_main_frame(self):
        """メインフレームの初期化"""
        # メインフレーム
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.grid(row=0, column=0, sticky=tk.NSEW)
        # ステータスフレーム
        self.status_frame = ttk.Frame(self.main_frame, borderwidth=1)
        self.status_frame.grid(row=1, column=0)
        self.prev_task_button = ttk.Button(self.status_frame, text='◀◀前の未処理画像', command=self.prev_task)
        self.prev_task_button.grid(row=0, column=0)
        self.prev_image_button = ttk.Button(self.status_frame, text='◀前の画像', command=self.prev_image)
        self.prev_image_button.grid(row=0, column=1)
        self.current_image_label = ttk.Label(self.status_frame, text='●.jpg')
        self.current_image_label.grid(row=0, column=2, sticky=tk.EW)
        self.next_image_button = ttk.Button(self.status_frame, text='次の画像▶', command=self.next_image)
        self.next_image_button.grid(row=0, column=3)
        self.next_task_button = ttk.Button(self.status_frame, text='次の未処理画像▶▶', command=self.next_task)
        self.next_task_button.grid(row=0, column=4)
        self.current_status_label = ttk.Label(self.status_frame, text='●枚／●枚　●%')
        self.current_status_label.grid(row=1, column=0, columnspan=5)

    def init_dish_frame(self):
        """皿フレームの初期化"""
        # 皿フレーム
        self.dish_frame = ttk.Frame(self.main_frame)
        self.dish_frame.grid(row=0, column=0, sticky=tk.NSEW)
        # 皿キャンバス
        self.dish_canvas = tk.Canvas(self.dish_frame, width=250, height=250, bd=0)
        self.dish_canvas.grid(row=0, column=0, sticky=tk.NSEW)
        # 皿メニューフレーム
        self.dish_menu = ttk.Frame(self.dish_frame, width=200, height=250)
        self.dish_menu.grid(row=0, column=1, sticky=tk.NSEW)
        # 二値化閾値
        self.binary_label = ttk.Label(self.dish_menu, text='二値化閾値')
        self.binary_label.grid(row=0, column=0)
        self.binary_entry = ttk.Entry(self.dish_menu, width=10)
        self.binary_entry.grid(row=0, column=1)
        # 最小半径
        self.min_radius_label = ttk.Label(self.dish_menu, text='最小半径')
        self.min_radius_label.grid(row=1, column=0)
        self.min_radius_entry = ttk.Entry(self.dish_menu, width=10)
        self.min_radius_entry.grid(row=1, column=1)
        self.min_radius_entry.insert(tk.END, '100')
        # 最大半径
        self.max_radius_label = ttk.Label(self.dish_menu, text='最大半径')
        self.max_radius_label.grid(row=2, column=0)
        self.max_radius_entry = ttk.Entry(self.dish_menu, width=10)
        self.max_radius_entry.grid(row=2, column=1)
        self.max_radius_entry.insert(tk.END, '112')
        # 領域抽出ボタン
        self.hough_button = ttk.Button(self.dish_menu, text='円形領域抽出', command=self.extract_hough)
        self.hough_button.grid(row=3, column=0)
        self.area_button = ttk.Button(self.dish_menu, text='任意領域抽出', command=self.extract_area)
        self.area_button.grid(row=3, column=1)
        # 皿保存ボタン
        self.save_dish_button = ttk.Button(self.dish_menu, text='皿領域保存', command=self.from_dish_to_mask)
        self.save_dish_button.grid(row=4, column=0, columnspan=2)

    def init_mask_frame(self):
        """マスクフレームの初期化"""
        # マスクフレーム
        self.mask_frame = ttk.Frame(self.main_frame)
        self.mask_frame.grid(row=0, column=0, sticky=tk.NSEW)
        # マスクキャンバス
        self.mask_canvas = tk.Canvas(self.mask_frame, width=250, height=250, bd=0)
        self.mask_canvas.grid(row=0, column=0, sticky=tk.NSEW)
        # マスクメニューフレーム
        self.mask_menu = ttk.Frame(self.mask_frame, width=200, height=250)
        self.mask_menu.grid(row=0, column=1, sticky=tk.NSEW)
        # レイヤーリストボックス
        self.layer_listbox = tk.Listbox(self.mask_menu, selectmode='extended')
        self.layer_listbox.grid(row=0, column=0)
        self.layer_scrollbar = tk.Scrollbar(self.mask_menu, command=self.layer_listbox.yview)
        self.layer_scrollbar.grid(row=0, column=1, sticky='ns')
        self.layer_listbox.config(yscrollcommand=self.layer_scrollbar.set)
        # 皿修正ボタン
        self.undo_dish_button = ttk.Button(self.mask_menu, text='皿領域修正', command=partial(self.from_mask_to_dish, True))
        self.undo_dish_button.grid(row=1, column=0)
        # マスク保存ボタン
        self.save_mask_button = ttk.Button(self.mask_menu, text='残留物領域保存', command=partial(self.from_mask_to_dish, False))
        self.save_mask_button.grid(row=1, column=1)

    def from_init_to_dish(self):
        """入力フレームから皿フレームへの遷移(開始ボタン押下)"""
        # TODO: 入力欄チェック
        for path in Path(self.input_entry.get()).glob('*.jpg'):
            self.image_paths.append(path)
        # TODO: 出力欄チェック
        self.output_path = Path(self.output_entry.get())
        self.output_image_path = self.output_path / 'image'
        self.output_dish_path = self.output_path / 'dish'
        self.output_mask_path = self.output_path / 'mask'
        self.output_image_path.mkdir(exist_ok=True)
        self.output_dish_path.mkdir(exist_ok=True)
        self.output_mask_path.mkdir(exist_ok=True)
        self.main_frame.tkraise()
        # ステータスの初期化
        self.num_images = len(self.image_paths)
        self.statuses = np.array([self.check_status(i) for i in range(self.num_images)], dtype=np.uint8)
        # ステータスに応じた画面遷移
        if self.statuses[self.index] == Application.STATUS_UNCOMPLETED:
            self.set_dish()
        else:
            self.set_mask()
        self.master.geometry('600x300')

    def from_dish_to_mask(self):
        """皿フレームからマスクフレームへの遷移(皿領域保存ボタン押下)"""
        self.save_dish()
        self.statuses[self.index] = Application.STATUS_DISH_COMPLETED
        self.set_mask()

    def from_mask_to_dish(self, undo=False):
        """マスクフレームから皿フレームへの遷移(皿領域修正、残留物領域保存ボタン押下)"""
        if undo:#皿領域修正ボタン押下
            # 現在の皿領域のやり直し
            self.statuses[self.index] = Application.STATUS_DISH_COMPLETED
        else:#残留物領域保存ボタン押下
            # 次の皿画像の取得
            self.save_mask()
            self.statuses[self.index] = Application.STATUS_MASK_COMPLETED
            self.count_up()
        self.set_dish()

    def set_dish(self):
        """現在の皿を表示"""
        # 画像の読み込み
        image = cv2.imread(str(self.image_paths[self.index]))
        # 保存した皿マスクの読み込み
        self.load_dish(image.shape)
        # 皿マスクの適用
        image[self.dish==0] = self.default_color
        # 画像の表示
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)
        self.dish_canvas.config(width=image.width, height=image.height)
        self.dish_canvas.photo = image_tk
        self.dish_image = self.dish_canvas.create_image(0, 0, anchor='nw', image=image_tk)
        # ステータスフレームに情報を表示
        self.set_status()
        self.dish_frame.tkraise()

    def set_mask(self):
        """現在のマスクを表示"""
        # 画像の読み込み
        image = cv2.imread(str(self.image_paths[self.index]))
        # 保存したマスクの読み込み
        self.load_dish(image.shape)
        # 皿マスクの適用
        image[self.dish==0] = self.default_color
        # クラスタリング
        #clusters = color_then_xy_clustering(image)
        clusters = color_clustering(image)
        print(np.unique(clusters))
        # 画像の表示
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)
        self.mask_canvas.config(width=image.width, height=image.height)
        self.mask_canvas.photo = image_tk
        self.mask_canvas.create_image(0, 0, anchor='nw', image=image_tk)
        #ここまででは皿マスクを行ったのみ

        # レイヤーの表示
        self.layers = []
        self.layer_listbox.delete(0, tk.END)
        for label in np.unique(clusters):
            self.layer_listbox.insert(tk.END, label)
            img = np.zeros((224, 224, 4), dtype=np.uint8)
            img[clusters==label] = [255, 0, 0, 64]
            img = Image.fromarray(img)
            print(np.array(img))
            #img = img.convert('RGBA')
            #print(np.array(img))
            img_tk = ImageTk.PhotoImage(img)
            self.mask_canvas.create_image(0, 0, tag=label, anchor='nw', image=img_tk)
            self.layers.append(img_tk)
            #print(cluster, len(cluster))
            break
        # ステータスフレームに情報を表示
        self.set_status()
        self.mask_frame.tkraise()

    def check_status(self, index=None):
        """指定インデックスの状態を確認"""
        # index未指定の場合は現在のindex
        if index is None:
            index = self.index
        # マスクの存在確認
        name = self.image_paths[index].stem
        dish_path = self.output_dish_path / f'{name}.png'
        mask_path = self.output_mask_path / f'{name}.png'
        if mask_path.exists():
            return Application.STATUS_MASK_COMPLETED
        elif dish_path.exists():
            return Application.STATUS_DISH_COMPLETED
        else:
            return Application.STATUS_UNCOMPLETED

    def set_status(self):
        """ステータスフレームに情報を表示"""
        name = self.image_paths[self.index].name
        num_done = np.sum(self.statuses == Application.STATUS_MASK_COMPLETED)
        progress = num_done / self.num_images
        check = '　'
        if self.statuses[self.index] == Application.STATUS_MASK_COMPLETED:
            check = '✓'
        self.current_image_label['text'] = f'{check} No.{self.index:>4}：{name}　'
        self.current_status_label['text'] = f'{num_done:04}枚／{self.num_images:>5}枚　{progress:.2%}'

    def prev_task(self):
        """現在より前の未完了画像を表示する"""
        for i in range(self.index):
            index = self.index - i - 1
            if self.statuses[index] == Application.STATUS_DISH_COMPLETED:
                self.index = index
                self.set_mask()
                break
            elif self.statuses[index] == Application.STATUS_UNCOMPLETED:
                self.index = index
                self.set_dish()
                break

    def next_task(self):
        """現在より次の未完了画像を表示する"""
        for i in range(len(self.statuses) - self.index - 1):
            index = self.index + i + 1
            if self.statuses[index] == Application.STATUS_DISH_COMPLETED:
                self.index = index
                self.set_mask()
                break
            elif self.statuses[index] == Application.STATUS_UNCOMPLETED:
                self.index = index
                self.set_dish()
                break

    def prev_image(self):
        """現在より前の画像を表示する"""
        self.count_down()
        if self.statuses[self.index] == Application.STATUS_UNCOMPLETED:
            self.set_dish()
        else:
            self.set_mask()

    def next_image(self):
        """現在より後の画像を表示する"""
        self.count_up()
        if self.statuses[self.index] == Application.STATUS_UNCOMPLETED:
            self.set_dish()
        else:
            self.set_mask()

    def extract_hough(self):
        """円形領域抽出"""
        #入力された最大・最小半径取得
        min_radius = int(self.min_radius_entry.get())
        max_radius = int(self.max_radius_entry.get())
        image = cv2.imread(str(self.image_paths[self.index]))
        #グレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #軽いぼかしを入れる
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        #openCVの機能を用いた円の検出。circlesには円の中心座標と半径が入る
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
        #print("circles=",circles)
        #cv2.imwrite('output.jpg', circles)
        if circles is not None:
            circles = np.uint16(np.around(circles))  # 整数化
            x, y, r = circles[0, 0]
            #今回の画像のサイズの画像imageを作成
            self.dish = np.zeros(image.shape[:-1], dtype=np.uint8)
            #円を描く
            self.dish = cv2.circle(self.dish, center=(x, y), radius=r, color=255, thickness=-1)
            cv2.imwrite('output.jpg', self.dish)
            cv2.imwrite('before_image.jpg', image)
            #円の領域の外側だけ色を塗る
            image[self.dish==0] = self.default_color
            cv2.imwrite('image.jpg', image)
            # TODO: GrabCut機能
            # TODO: GrabCutの加工中マスク保持
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #青みがかった画像に変換
            cv2.imwrite('image_rgb.jpg', image)
            image = Image.fromarray(image)
            image_tk = ImageTk.PhotoImage(image)
            #抽出した画像の表示
            self.dish_canvas.config(width=image.width, height=image.height)
            self.dish_canvas.photo = image_tk
            self.dish_canvas.itemconfig(self.dish_image, image=self.dish_canvas.photo)

    def extract_area(self):
        """任意領域抽出"""
        image = cv2.imread(str(self.image_paths[self.index]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary_th = self.binary_entry.get()
        # TODO: cv2.adaptiveThreshold
        # TODO: cv2.THRESH_TRIANGLE
        if binary_th:
            # 二値化閾値の入力がある場合、入力値を使用
            binary_th = int(binary_th)
            retval, binary = cv2.threshold(gray, binary_th, 255, cv2.THRESH_BINARY)
        else:
            # 二値化閾値の入力がない場合、大津の二値化を使用
            retval, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.binary_entry.delete(0, tk.END)
            self.binary_entry.insert(tk.END, int(retval))
        # 領域抽出
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 最大領域を採用
        areas = np.array([cv2.contourArea(contour) for contour in contours])
        index = np.argmax(areas)
        # 皿マスク生成と描画
        self.dish = np.zeros(image.shape[:-1], dtype=np.uint8)
        self.dish = cv2.drawContours(self.dish, contours, index, 255, -1)
        image[self.dish==0] = self.default_color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)
        self.dish_canvas.config(width=image.width, height=image.height)
        self.dish_canvas.photo = image_tk
        self.dish_canvas.itemconfig(self.dish_image, image=self.dish_canvas.photo)

root = tk.Tk()
app = Application(master=root)
app.mainloop()