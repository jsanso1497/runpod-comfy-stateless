"""Run the consolidated native integration check against the CI server only."""
from checks.native_check import main
import asyncio
if __name__=='__main__':
    asyncio.run(main())
