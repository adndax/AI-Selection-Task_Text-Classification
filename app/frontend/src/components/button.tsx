'use client';

interface ButtonProps {
  children: React.ReactNode;
  variant: 'red' | 'green';
  onClick?: () => void;
  className?: string;
}

export default function Button({ children, variant, onClick, className = "" }: ButtonProps) {
  const baseStyles = "cursor-pointer w-full px-14 py-2.5 text-white font-poppins font-medium text-lg rounded-lg transition-all duration-200";
  
  const variantStyles = {
    red: "bg-red-600 hover:bg-red-700",
    green: "bg-green-600 hover:bg-green-700"
  };

  return (
    <button
      className={`${baseStyles} ${variantStyles[variant]} ${className}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}