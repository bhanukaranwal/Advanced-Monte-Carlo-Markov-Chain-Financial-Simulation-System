-- Initial database setup for MCMF system
-- Create database and user (if not exists)

-- Create extension for UUID support
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tables for market data
CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open_price DECIMAL(15,4),
    high_price DECIMAL(15,4),
    low_price DECIMAL(15,4),
    close_price DECIMAL(15,4) NOT NULL,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index for efficient querying
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);

-- Create table for simulation results
CREATE TABLE IF NOT EXISTS simulation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    simulation_type VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,
    results JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100)
);

-- Create table for backtest results
CREATE TABLE IF NOT EXISTS backtest_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_name VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15,2) NOT NULL,
    final_capital DECIMAL(15,2) NOT NULL,
    total_return DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    num_trades INTEGER,
    parameters JSONB,
    detailed_results JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create table for portfolio positions
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(15,4) NOT NULL,
    average_price DECIMAL(15,4),
    market_value DECIMAL(15,2),
    unrealized_pnl DECIMAL(15,2),
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Create table for risk metrics
CREATE TABLE IF NOT EXISTS risk_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL,
    metric_date DATE NOT NULL,
    var_95 DECIMAL(15,4),
    var_99 DECIMAL(15,4),
    expected_shortfall_95 DECIMAL(15,4),
    volatility DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create table for system logs
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    level VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    module VARCHAR(100),
    function_name VARCHAR(100),
    line_number INTEGER,
    timestamp TIMESTAMP DEFAULT NOW(),
    additional_data JSONB
);

-- Create table for user sessions (if authentication is implemented)
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100) NOT NULL,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_simulation_results_created_at ON simulation_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy ON backtest_results(strategy_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_portfolio ON portfolio_positions(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_portfolio_date ON risk_metrics(portfolio_id, metric_date DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);

-- Create a function to clean old logs (older than 30 days)
CREATE OR REPLACE FUNCTION clean_old_logs()
RETURNS void AS $$
BEGIN
    DELETE FROM system_logs WHERE timestamp < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- Create a function to calculate portfolio statistics
CREATE OR REPLACE FUNCTION calculate_portfolio_stats(p_portfolio_id UUID)
RETURNS TABLE (
    total_value DECIMAL(15,2),
    total_pnl DECIMAL(15,2),
    num_positions INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(SUM(market_value), 0) as total_value,
        COALESCE(SUM(unrealized_pnl), 0) as total_pnl,
        COUNT(*)::INTEGER as num_positions
    FROM portfolio_positions 
    WHERE portfolio_id = p_portfolio_id;
END;
$$ LANGUAGE plpgsql;

-- Insert some sample data for testing
INSERT INTO market_data (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
VALUES 
    ('AAPL', '2023-01-01 09:30:00', 150.00, 152.50, 149.00, 151.25, 1000000),
    ('GOOGL', '2023-01-01 09:30:00', 2800.00, 2825.00, 2795.00, 2810.50, 500000),
    ('MSFT', '2023-01-01 09:30:00', 300.00, 302.75, 299.50, 301.85, 750000)
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO mcmf_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO mcmf_user;

-- Create notification function for real-time updates
CREATE OR REPLACE FUNCTION notify_market_data_change()
RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('market_data_change', 
        json_build_object('symbol', NEW.symbol, 'price', NEW.close_price)::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for market data notifications
DROP TRIGGER IF EXISTS market_data_notify_trigger ON market_data;
CREATE TRIGGER market_data_notify_trigger
    AFTER INSERT OR UPDATE ON market_data
    FOR EACH ROW
    EXECUTE FUNCTION notify_market_data_change();

COMMIT;
