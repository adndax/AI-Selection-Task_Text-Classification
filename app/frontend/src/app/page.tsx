'use client';

import Button from '@/components/button';
import Image from 'next/image'
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();

  const handleStartClick = () => {
    router.push('/start');
  }

  const handleAboutClick = () => {
    router.push('/about');
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4" 
         style={{ background: 'linear-gradient(135deg, #2D1B69 0%, #11052C 100%)' }}>
      
      {/* Logo */}
      <div className="mb-8">
        <Image
          src="/logo.png"
          alt="FeelOut Logo"
          width={180}
          height={180}
        />
      </div>

      {/* Title */}
      <h1 className="text-4xl font-irish-grover mb-10 font-bold">
        <span className="text-yellow-400">Feel</span>
        <span className="text-blue-400">Out</span>
      </h1>

      {/* Buttons */}
      <div className="flex flex-col gap-3">
        <Button variant="red" onClick={handleStartClick}>
          Start
        </Button>
        <Button variant="green" onClick={handleAboutClick}>
          About
        </Button>
      </div>
    </div>
  );
}