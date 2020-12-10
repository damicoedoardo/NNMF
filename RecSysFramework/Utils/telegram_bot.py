import telepot
import threading

"""
To create a new private channel, you will need 2 things:
1) A token bot: you can create by BotFather inside telegram. From the bot, you have to get the 'token'
    following the on-screen commands
2) A chat id: start the bot on telegram, send any message to the bot (important!) and then go to this address:
    https://api.telegram.org/bot<yourtoken>/getUpdates
    (replace <yourtoken> with the one got at the previous step)
    Copy the id from of the chat object, ex: ... , "chat": {"id": 123456789, "first_name": ...
"""
#chat_id = -1001481984580
#token = '819065046:AAFee77GqSpq8XBzmEnAMybLqOHuy6PJ_bg'

# stores chat_id and tokens
accounts = {
    'gabbo': (361500321, '800854524:AAGUxIYNxcVHKyjbiQk_SbU-jWj1-3lSpEA'),
    'eg': (-1001386765318, '675236794:AAEpSgQ44Ncs1a8nh_uvc8AXaWvspI6pz1U')
    # <insert your chat_id and token here>
}

# caches created bots per account
bots = {account: None for account in accounts.keys()}


def get_bot(account):
    """ Get or create a new bot and cache it in the dictionary.
        Return bot and chat_id
    """
    if account not in accounts:
        print('Invalid telegram bot account!')
        return None, None

    chat_id, token = accounts[account]
    if bots[account] is None:
        bots[account] = telepot.Bot(token)

    return bots[account], chat_id


def send_message(message, account='eg'):
    # bot tries to re-send in case an exception is raised
    try:
        bot, chat_id = get_bot(account)
        bot.sendMessage(chat_id=chat_id, text=message)
    except:
        threading.Timer(5, send_message, [message, account]).start()
