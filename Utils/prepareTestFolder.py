import os
import shutil

train_path = './../Data/train/'
test_path = './../Data/test/'

def list_files(path):
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")

# list of images, that have to be moved to test folder
image_lists = {
    'A400M': [
        '6deae45660f7f28c6faa4e47d15500a3_2.jpg', 
        '9c68108b79fa87e30de6ba6609c97d22_0.jpg', 
        '677e0eb8e32a5074bfd8130e9679f92e_0.jpg', 
        '75564e9f3f17c7874e5260a4a3cec8ee_2.jpg', 
        '254388ed068c3e116c368f2b22343ec5_0.jpg', 
        '972038762c34b66caaef598bdd61a48b_0.jpg', 
        'a6df7d22743528a49e815e81cc5e96c5_0.jpg', 
        'b666267f2b0cd31cf358778b1158ec81_0.jpg', 
        'cb00888189e2a56640e4a98b40c9d909_1.jpg', 
        'd9c7496d22a0c903157077997495541d_2.jpg'],
    'C130': [
        '3d23841d25007c85d13b2e663321ad39_4.jpg', 
        '8c522c3774c3a64720341b48165210aa_0.jpg', 
        '47fd84d4bfc5ac09f762a999756b7bc3_0.jpg', 
        '341e68a0c63b6b7681233c4324fbca79_0.jpg', 
        '893ed36b443cc9b0e7b244ced87a65c9_0.jpg', 
        '18121db1452fd759adb4f51a01adb2ab_0.jpg', 
        '031726327f5f51788fbf03a1dc69527d_0.jpg', 
        'a3dccd36c4a68de45520c57517464ed2_0.jpg', 
        'b6e8a7954e3e906f6779f40caa082bb3_0.jpg', 
        'bff74276f9195233b82fe8780d656f10_2.jpg'],
    'Su57': [
        '2a735ee961ce7fabe6c6217f3a3ef53b_0.jpg', 
        '6ab2a802b87934018935a4388aa9176b_0.jpg', 
        '30aaf62a22cd2dde0aa90614da20172c_0.jpg', 
        '73cea06069b9d1f0745a3de4d336bd90_1.jpg', 
        '495fada28cab24907fda48ac6d964007_0.jpg', 
        '86011f1f186389d7bebf2c6357106160_0.jpg', 
        'c2b8ce926662b1f8af69f89ac1256c01_0.jpg', 
        'cd152d7c293d3cbd44fd323535610975_0.jpg', 
        'cf203d5421aef5176ec086730e3cbada_0.jpg', 
        'dbc90c8aae7e2ccd83915a314a95252e_0.jpg'],
    'Tu160': [
        '0c02addad95322392e327032a3b0d2b2_0.jpg', 
        '2cbe25b281bab1a4f3a7029c44bac3c5_0.jpg', 
        '4f6335afb880904ed5ebae7ed55fd81b_0.jpg', 
        '6c619d7818b1eeefda7a3e79c5795347_0.jpg', 
        '8d6141670cc7a7894a789a91d461413c_3.jpg', 
        '28b5bff3e86f8dae8d1333279b4fba31_0.jpg', 
        '50f7a0b92a276b000c0fe6905d1d3614_0.jpg', 
        '82fb466dc9a61953bc4e1267ab3691aa_0.jpg', 
        '697ba3e167a92fead0b4514bd14b9f5c_0.jpg', 
        'cf0c2a3be99325f5bae7118c525be8b0_0.jpg']
}

for folder in image_lists:
    folder_path_train = os.path.join(train_path, folder)
    folder_path_test = os.path.join(test_path, folder)

    if not os.path.exists(folder_path_test):
        os.makedirs(folder_path_test)

    for image in image_lists[folder]:
        file_path_train = os.path.join(folder_path_train, image)
        file_path_test = os.path.join(folder_path_test, image)
        shutil.move(file_path_train, file_path_test)

list_files(test_path)
