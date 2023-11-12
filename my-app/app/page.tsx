import Image from 'next/image'
import GetApi from './components/getApi'


const Home = () => {
  
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <GetApi />
    </main>
  )
}

export default Home