import React, { useState } from "react";
import { useNavigate } from "react-router";
import { useAuth } from "../../contexts/AuthContext";
import Input from "../ui/Input";
import Button from "../ui/Button";
import Card from "../ui/Card";
import "./SignupForm.css";

const SignupForm: React.FC = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { signup, error, clearError } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (isSubmitting) return;
    
    if (password !== confirmPassword) {
      // You might want to show this error in a similar way to Firebase errors
      alert("Passwords do not match");
      return;
    }
    
    setIsSubmitting(true);
    clearError();

    try {
      await signup({ email, password, confirmPassword });
      navigate("/dashboard");
    } catch (error) {
      // Error is handled by the AuthContext
      console.error("Signup failed:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card className="signup-card">
      <form className="signup-form" onSubmit={handleSubmit}>
        <h2>Sign Up</h2>
        
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
        <Input
          type="password"
          placeholder="Confirm Password"
          value={confirmPassword}
          onChange={e => setConfirmPassword(e.target.value)}
          required
          disabled={isSubmitting}
        />
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Creating Account..." : "Create Account"}
        </Button>
        
        <div style={{ textAlign: 'center', marginTop: '0.5rem' }}>
          <span style={{ color: '#e0e0e0', fontSize: '1rem' }}>
            Already have an account?{' '}
            <a href="/login" style={{ color: '#fff', textDecoration: 'underline', opacity: 0.85 }}>Sign in</a>
          </span>
        </div>
      </form>
    </Card>
  );
};

export default SignupForm;
