import React, { useState } from "react";
import { useNavigate } from "react-router";
import { useAuth } from "../../contexts/AuthContext";
import Input from "../ui/Input";
import Button from "../ui/Button";
import Card from "../ui/Card";
import "./LoginForm.css";

const LoginForm: React.FC = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { login, error, clearError } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (isSubmitting) return;
    
    setIsSubmitting(true);
    clearError();

    try {
      await login({ email, password });
      navigate("/dashboard");
    } catch (error) {
      // Error is handled by the AuthContext
      console.error("Login failed:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card className="login-card">
      <form className="login-form" onSubmit={handleSubmit}>
        <h2>Login</h2>
        
        {error && (
          <div style={{ 
            color: '#ff6b6b', 
            backgroundColor: 'rgba(255, 107, 107, 0.1)',
            border: '1px solid rgba(255, 107, 107, 0.3)',
            borderRadius: '4px',
            padding: '0.75rem',
            marginBottom: '1rem',
            fontSize: '0.875rem'
          }}>
            {error}
          </div>
        )}
        
        <Input
          type="email"
          placeholder="Email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
          disabled={isSubmitting}
        />
        <Input
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
          disabled={isSubmitting}
        />
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Signing In..." : "Sign In"}
        </Button>
        <div style={{ textAlign: 'center', marginTop: '0.5rem' }}>
          <span style={{ color: '#e0e0e0', fontSize: '1rem' }}>
            Don't have an account?{' '}
            <a href="/signup" style={{ color: '#fff', textDecoration: 'underline', opacity: 0.85 }}>Sign up</a>
          </span>
        </div>
      </form>
    </Card>
  );
};

export default LoginForm;
