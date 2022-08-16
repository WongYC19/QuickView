# EquityAnalyzer
To find investment opportunity and trading signal through fundamental analysis and technical analysis in Bursa Malaysia (KLSE)

# Issue Tracker
<ol>
    <li> <b> Fundamental Analysis </b>
    <ol>
    <br/>
        <li> To collect financial statement for both Annual and Quarter frequency
        <li> To generate the overview of company performance using statistical score:
        <ol>
            <li> Altman Z Score (Simple Credit Strength test)
            <li> Ohlson O Score (Extended from Z-Score, Likelihood to default in 2 years)
            <li> Beneish M Score (8) - Profit Manipulation 
            <li> Piotroski F Score (Determine strength of a firm's financial position, avoid value traps)
            <li> EBIT / EV (Alternative to PE ratio, take into account of debt)
            <li> Net Cash (Net cash position of a company)
            <li> Market Capitalization (Market value of company outstanding shares)
            <li> 52W Range vs Current Price (Position of share price over last 52 weeks) 
        </ol>        
        <li> To collect the commodity price and foreign exchange        
            <ol>
                <li> Rubberwood
                <li> Resin
                <li> Gold
                <li> Lumber
                <li> USD/MYR
                <li> Baltic dry
                <li> Crude oil
                <li> Palm oil
                <li> Lumber
                <li> Sugar
                <li> Poultry
                <li> Corn
                <li> Steel
                <li> Fearix
            </ol>          
        <li> To update the existing data file rather than replacing it
    </ol>    
    <br/>
    <li> <b> Technical Analysis </b> <br/>
    <ol>
    <br/>
        <li> To collect historical data from yahoo API
        <li> Create and define technical indicator        
        <li> Modelling        
            <ol>
                <li> Classification - Trading signal (Buy or sell?)
                <li> Regression - Entry Price / Stop loss (Buy/Sell at what price?) <br/>
            </ol>
    </ol> 
    <br/>
    <li> <b> Natural Language Processing (NLP) - TBC </b>    
    <ol>
    <br/>
        <li> Footnote preparation / Remarks
            - To collect and summarize information from quarter report / analyst report
        <li> News / Forum
            - Sentiment Analysis <br/>
    </ol>       
    <br/>
    <li> <b> User Interface </b>    
    <ol>
    <br/>        
        <li> Table (Statement, historical share price)
        <li> Graph (Historical Share price and the technical indicator)
        <li> Trading signal (Search and filter function)
    </ol>
</ol>  
