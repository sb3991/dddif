import re

# 입력 파일과 출력 파일의 이름을 지정합니다.
input_file = "/home/a5t11/ICML_Symbolic/Repo/Data/label-prompt/imagenet_readable.txt"
output_file = "sorted_imagenet_classes.txt"

# 데이터를 저장할 리스트를 초기화합니다.
parsed_data = []

# 입력 파일을 읽고 데이터를 파싱합니다.
with open(input_file, "r") as file:
    for line in file:
        match = re.match(r'(n\d+)\s+(\d+)\s+(.+)', line.strip())
        if match:
            n_id, index, class_name = match.groups()
            parsed_data.append((n_id, int(index), class_name))

# n_id를 기준으로 데이터를 정렬합니다.
sorted_data = sorted(parsed_data, key=lambda x: x[0])

# 정렬된 데이터를 새 파일에 저장합니다.
with open(output_file, "w") as file:
    for n_id, index, class_name in sorted_data:
        file.write(f"{n_id} {index} {class_name}\n")

print(f"데이터가 '{output_file}' 파일로 저장되었습니다.")

# 저장된 데이터 확인 (선택사항)
print(f"\n저장된 데이터의 처음 5줄:")
with open(output_file, "r") as file:
    for _ in range(5):
        print(file.readline().strip())