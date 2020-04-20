import telebot
import constant
import number_recognizer

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True


bot = telebot.TeleBot(constant.token)


def process_photo_message(message):
    print('message.photo =', message.photo)
    fileID = message.photo[-1].file_id
    print('fileID =', fileID)
    file = bot.get_file(fileID)
    print('file.file_path =', file.file_path)
    downloaded_file = bot.download_file(file.file_path)
    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)
    number_length, number = number_recognizer.recognize(downloaded_file)
    str()

    return 'My prediction is -\nNumber length: {}\nNumber: {}'.format(number_length, number)


@bot.message_handler(content_types=['photo'])
def photo(message):
    recognized_num = process_photo_message(message)
    bot.send_message(message.chat.id, recognized_num)


bot.polling()


