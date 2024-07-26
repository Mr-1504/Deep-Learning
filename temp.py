import os
import random
import shutil


def get_random_files_from_directory(source_directory, destination_directory, number_of_files):
    # Lấy danh sách tất cả các tệp tin trong thư mục nguồn
    all_files = [os.path.join(source_directory, f) for f in os.listdir(source_directory) if
                 os.path.isfile(os.path.join(source_directory, f))]

    # Kiểm tra nếu số lượng tệp tin ít hơn số lượng yêu cầu
    if len(all_files) < number_of_files:
        raise ValueError("Số lượng tệp tin trong thư mục ít hơn số lượng yêu cầu.")

    # Lấy ngẫu nhiên số lượng tệp tin được yêu cầu
    random_files = random.sample(all_files, number_of_files)

    # Đảm bảo thư mục đích tồn tại, nếu không thì tạo mới
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Sao chép các tệp tin được chọn vào thư mục đích
    for file in random_files:
        shutil.copy(file, destination_directory)

    return random_files


# Ví dụ sử dụng
source_directory = 'D:\\Code\\Vigilant-VGG16\\Brain\\resoucre\\data\\Normal'
destination_directory = 'D:\\user\\Desktop\\temp'
number_of_files = 950

try:
    random_files = get_random_files_from_directory(source_directory, destination_directory, number_of_files)
    print(f"Đã sao chép {len(random_files)} tệp tin vào thư mục {destination_directory}")
except ValueError as e:
    print(e)
