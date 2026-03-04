import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from dotenv import load_dotenv
from search_engine import SearchEngine

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")


bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)

dp = Dispatcher()

engine = SearchEngine()


@dp.message(CommandStart())
async def start(m: types.Message):
    await m.answer("Salom! Savolingizni yuboring. Men kitoblardan tegishli parchani topib beraman.")


@dp.message()
async def handle_query(m: types.Message):
    q = (m.text or "").strip()
    if not q:
        return


    msg = await m.answer("Qidiryapman...")


    result = await asyncio.to_thread(engine.search, q)

    if not result:
        await msg.edit_text("Javob topilmadi.")
        return


    texts = []
    for r in result[:3]:
        texts.append(
            f"<b>Kitob nomi:</b> {r['book_title']}\n"
            f"<b>Sallavha:</b> {r.get('heading','NO_HEADING')}\n"
            f"<b>Malumot:</b> {r['information']}\n"
        )

    await msg.edit_text("\n\n".join(texts))


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
