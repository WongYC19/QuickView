import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

SVG_CODES = {
    "shariah": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.97 1 14.99 14.93"><path d="M8.468 1.001c-.08 0-.158.03-.219.09L6.252 3.08l-2.865-.006a.32.32 0 0 0-.22.09.323.323 0 0 0-.094.224l.006 2.85L1.06 8.245a.309.309 0 0 0-.09.222c0 .084.031.161.09.22L3.09 10.71l.006 2.818c0 .171.14.311.311.311l2.833.006 2.007 2a.336.336 0 0 0 .22.088.324.324 0 0 0 .221-.09l1.997-1.988 2.865.005a.309.309 0 0 0 .311-.311l-.006-2.848 2.019-2.01a.32.32 0 0 0 0-.445l-2.029-2.022-.006-2.818a.312.312 0 0 0-.31-.311l-2.834-.006-2.007-1.997A.309.309 0 0 0 8.468 1Zm-.001.75 1.878 1.87c.06.059.137.09.218.09l2.65.006.007 2.638a.32.32 0 0 0 .09.22l1.903 1.892-1.891 1.882a.296.296 0 0 0-.09.22l.006 2.666-2.681-.005c-.081 0-.162.03-.22.09l-1.87 1.863-1.88-1.87a.305.305 0 0 0-.217-.09l-2.65-.006-.007-2.637a.322.322 0 0 0-.09-.222L1.72 8.468 3.61 6.586a.295.295 0 0 0 .09-.221l-.003-2.666 2.681.006c.081 0 .162-.031.221-.09zm1.978 3.145a.314.314 0 0 0-.28.174l-.323.654-.722.105a.31.31 0 0 0-.252.212.315.315 0 0 0 .078.32l.522.507-.125.719a.314.314 0 0 0 .125.304.317.317 0 0 0 .327.025l.644-.339.643.34a.306.306 0 0 0 .147.034.312.312 0 0 0 .308-.364l-.125-.719.523-.507a.309.309 0 0 0 .078-.317.286.286 0 0 0-.243-.215l-.722-.105-.323-.654a.315.315 0 0 0-.28-.174ZM7.75 4.9a.316.316 0 0 0-.095.008 3.593 3.593 0 0 0-1.696.952 3.623 3.623 0 0 0-.003 5.117 3.608 3.608 0 0 0 2.56 1.058 3.608 3.608 0 0 0 3.512-2.754.31.31 0 0 0-.128-.33v.001a.31.31 0 0 0-.354.006 2.549 2.549 0 0 1-3.297-.273 2.548 2.548 0 0 1-.27-3.295.314.314 0 0 0 .005-.354.302.302 0 0 0-.234-.136zm-.683.895a3.167 3.167 0 0 0 .74 3.328 3.163 3.163 0 0 0 3.332.743 3 3 0 0 1-4.744.67 3 3 0 0 1 .672-4.741zm3.378.112.115.233a.314.314 0 0 0 .233.171l.258.038-.18.186a.31.31 0 0 0-.09.274l.044.258-.23-.121a.347.347 0 0 0-.147-.034c-.05 0-.1.012-.146.034l-.23.121.043-.258a.308.308 0 0 0-.09-.277L9.84 6.35l.257-.038a.304.304 0 0 0 .234-.17z"></path></svg>',
    "sector": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><path style="line-height:normal;text-indent:0;text-align:start;text-decoration-line:none;text-decoration-style:solid;text-decoration-color:#000;text-transform:none;block-progression:tb;isolation:auto;mix-blend-mode:normal" d="M24 0C10.751 0 0 10.751 0 24s10.751 24 24 24 24-10.751 24-24S37.249 0 24 0zm0 1c12.708 0 23 10.291 23 23S36.708 47 24 47 1 36.709 1 24 11.292 1 24 1zm5 10a2.508 2.508 0 0 0-2.5 2.5c0 .694.289 1.323.75 1.777l-2.307 3.846A3.97 3.97 0 0 0 24 19c-1.197 0-2.262.54-2.996 1.379L20.5 20l-.6.8.543.409A3.946 3.946 0 0 0 20 23c0 .64.166 1.237.436 1.775l-5.694 3.795A2.968 2.968 0 0 0 13 28c-1.65 0-3 1.35-3 3 0 1.651 1.35 3 3 3s3-1.349 3-3c0-.634-.2-1.222-.54-1.707l5.532-3.688C21.727 26.452 22.796 27 24 27c.644 0 1.245-.169 1.785-.441l.389.546.814-.582-.375-.521C27.456 25.268 28 24.201 28 23c0-.021-.005-.04-.006-.06l6.338-.846A2 2 0 0 0 36 23c1.099 0 2-.901 2-2s-.901-2-2-2-2 .901-2 2c0 .044.01.085.014.129l-6.172.822a3.996 3.996 0 0 0-1.955-2.455l2.205-3.674c.282.112.587.178.908.178 1.375 0 2.5-1.125 2.5-2.5S30.375 11 29 11zm0 1c.834 0 1.5.666 1.5 1.5S29.834 15 29 15s-1.5-.666-1.5-1.5.666-1.5 1.5-1.5zm-13 3c-1.099 0-2 .901-2 2s.901 2 2 2c.395 0 .761-.12 1.072-.32l.428.32.6-.799-.348-.262c.153-.282.248-.598.248-.939 0-1.099-.901-2-2-2zm0 1c.558 0 1 .442 1 1s-.442 1-1 1-1-.442-1-1 .442-1 1-1zm2.9 2.8-.6.802.8.6.6-.802-.8-.6zM24 20c1.663 0 3 1.337 3 3s-1.337 3-3 3-3-1.337-3-3 1.337-3 3-3zm12 0c.558 0 1 .442 1 1s-.442 1-1 1-1-.442-1-1 .442-1 1-1zm-8.43 7.338-.814.582.58.812.814-.58-.58-.814zm1.162 1.627-.814.582.582.812.813-.58-.58-.814zM13 29c1.11 0 2 .89 2 2 0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2zm16.895 1.592-.815.582.582.814.813-.582-.58-.814zM32.5 32a2.47 2.47 0 0 0-1.324.389l-.12-.168-.814.58.203.283A2.48 2.48 0 0 0 30 34.5c0 1.375 1.125 2.5 2.5 2.5s2.5-1.125 2.5-2.5-1.125-2.5-2.5-2.5zm0 1c.834 0 1.5.666 1.5 1.5s-.666 1.5-1.5 1.5-1.5-.666-1.5-1.5.666-1.5 1.5-1.5z" color="#000" font-family="sans-serif" font-weight="400" overflow="visible"></path></svg>',
    "industry": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="5 5 90 90"><path d="M18.4 69.6c.4.4.6.9.6 1.4v.4c0 .1-.1.3-.1.4 0 .1-.1.2-.2.3l-.3.3c-.4.4-.9.6-1.4.6s-1-.2-1.4-.6c-.1-.1-.2-.2-.2-.3-.1-.1-.1-.2-.2-.3-.1-.1-.1-.2-.1-.4V71c0-.5.2-1 .6-1.4.1-.1.2-.2.3-.2.1-.1.2-.1.3-.2.1 0 .2-.1.4-.1.6-.2 1.3 0 1.7.5zm8.2-.6c-.1 0-.2.1-.4.1-.1.1-.2.1-.3.2-.1.1-.2.2-.3.2-.4.5-.6 1-.6 1.5v.4c0 .1.1.3.1.4 0 .1.1.2.2.3.1.1.1.2.2.3.5.4 1 .6 1.5.6s1-.2 1.4-.6c.1-.1.2-.2.2-.3.1-.1.1-.2.2-.3.1-.1.1-.2.1-.4V71c0-.5-.2-1-.6-1.4-.3-.5-1-.7-1.7-.6zm10 0c-.1 0-.2.1-.4.1-.1.1-.2.1-.3.2-.1.1-.2.2-.3.2-.4.5-.6 1-.6 1.5v.4c0 .1.1.3.1.4 0 .1.1.2.2.3.1.1.1.2.2.3.1.1.2.2.3.2.1.1.2.1.3.2.1 0 .2.1.4.1h.4c.5 0 1-.2 1.4-.6.1-.1.2-.2.2-.3.1-.1.1-.2.2-.3.1-.1.1-.2.1-.4v-.4c0-.5-.2-1-.6-1.4-.2-.4-.9-.6-1.6-.5zM15 27c0-6 4.4-11 10.2-11.8C26 9.4 31 5 37 5h16c1.1 0 2 .9 2 2 0 6-4.4 11-10.2 11.8C44 24.6 39 29 33 29H17c-1.1 0-2-.9-2-2zm14.2-12H43c3.7 0 6.9-2.5 7.8-6H37c-3.7 0-6.9 2.5-7.8 6zm-10 10H33c3.7 0 6.9-2.5 7.8-6H27c-3.7 0-6.9 2.5-7.8 6zM89 56.4V89c0 3.3-2.7 6-6 6H51c-1.5 0-2.9-.6-4-1.5-1.1 1-2.5 1.5-4 1.5H11c-3.3 0-6-2.7-6-6V66c0-2.7 1.7-5 4.3-5.8l5.7-1.7V42c0-3.9 3.1-7 7-7s7 3.1 7 7v12.3l12.3-3.7c1.8-.6 3.8-.2 5.3.9s2.4 2.9 2.4 4.8v4c.1 0 .2-.1.3-.1l5.7-1.7V42c0-3.9 3.1-7 7-7s7 3.1 7 7v12.3l12.3-3.7c1.8-.6 3.8-.2 5.3.9 1.5 1.2 2.4 3 2.4 4.9zm-70 .9 6-1.8V42c0-1.7-1.3-3-3-3s-3 1.3-3 3v15.3zM35 83H19v8h16v-8zm10-26.6c0-.6-.3-1.2-.8-1.6-.3-.3-.8-.4-1.2-.4-.2 0-.4 0-.6.1l-32 9.6c-.8.2-1.4 1-1.4 1.9v23c0 1.1.9 2 2 2h4V81c0-1.1.9-2 2-2h20c1.1 0 2 .9 2 2v10h4c1.1 0 2-.9 2-2V56.4zm14 .9 6-1.8V42c0-1.7-1.3-3-3-3s-3 1.3-3 3v15.3zM75 83H59v8h16v-8zm10-26.6c0-.6-.3-1.2-.8-1.6-.3-.3-.8-.4-1.2-.4-.2 0-.4 0-.6.1l-32 9.6c-.9.3-1.4 1-1.4 1.9v23c0 1.1.9 2 2 2h4V81c0-1.1.9-2 2-2h20c1.1 0 2 .9 2 2v10h4c1.1 0 2-.9 2-2V56.4zM55.6 69.6c-.4.4-.6.9-.6 1.4v.4c0 .1.1.3.1.4 0 .1.1.2.2.3l.3.3c.2.2.4.3.6.4.3.2.5.2.8.2h.4c.1 0 .3-.1.4-.1.1 0 .2-.1.3-.2.1-.1.2-.1.3-.2.1-.1.2-.2.2-.3.1-.1.1-.2.2-.3.1-.1.1-.2.1-.4v-.4c0-.5-.2-1-.6-1.4-.6-.9-2-.9-2.7-.1zm10 0c-.4.4-.6.9-.6 1.4v.4c0 .1.1.3.1.4 0 .1.1.2.2.3l.3.3c.2.2.4.3.6.4.3.2.5.2.8.2h.4c.1 0 .3-.1.4-.1.1 0 .2-.1.3-.2.1-.1.2-.1.3-.2.1-.1.2-.2.2-.3.1-.1.1-.2.2-.3.1-.1.1-.2.1-.4v-.4c0-.5-.2-1-.6-1.4-.6-.9-2-.9-2.7-.1zm13.2.6c0-.1-.1-.2-.2-.3-.1-.1-.1-.2-.2-.3-.1-.1-.2-.2-.3-.2s-.2-.1-.3-.2c-.1 0-.2-.1-.4-.1-.7-.1-1.3.1-1.8.5-.1.1-.2.2-.2.3-.1.1-.1.2-.2.3-.1.1-.1.2-.1.4v.4c0 .5.2 1 .6 1.4.3.4.8.6 1.3.6s1-.2 1.4-.6c.4-.4.6-.9.6-1.4v-.4c-.1-.1-.1-.3-.2-.4zM95 7c0 6-4.4 11-10.2 11.8C84 24.6 79 29 73 29H57c-1.1 0-2-.9-2-2 0-6 4.4-11 10.2-11.8C66 9.4 71 5 77 5h16c1.1 0 2 .9 2 2zM80.8 19H67c-3.7 0-6.9 2.5-7.8 6H73c3.7 0 6.9-2.5 7.8-6zm10-10H77c-3.7 0-6.9 2.5-7.8 6H83c3.7 0 6.9-2.5 7.8-6z"></path></svg>',
}
    
def render_icon(icon_name, width=50, height=50, caption=None, **kwargs):
    svg_code = SVG_CODES.get(icon_name)
    
    if svg_code is None:
        return st.write("Not found.")  
    
    caption = caption or ""
            
    svg_code = f"""
        <figure style="font-size: 1.2em; display: flex; flex-direction: row; justify-content: flex-start; align-items: center; margin: 0;">
            <p> 
                {svg_code.replace(" viewBox=", f"width={width} height={height} viewBox=")}                                    
                <figcaption style="margin-left: 20px; font-weight: 600;">{caption}</figcaption>
            </p>
        </figure>
    """
    
    st.write(svg_code, **kwargs, unsafe_allow_html=True)
    
def render_candlestick(share_price, name=None, use_container_width=True):
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_width=[0.2, 0.7]
    )
    
    candlestick = { 
        'x': share_price.index,
        'open': share_price.Open,
        'close': share_price.Close,
        'high': share_price.High,
        'low': share_price.Low,
        'type': 'candlestick',
        "showlegend": False,
        'name': name,
        # "increasing_line_color": 'white', 
        # "decreasing_line_color": 'black',
    }
    
    volume = {
        "x": share_price.index,
        "y": share_price.Volume,
        "type": "bar",
        "showlegend": False,
        "name": "Volume",
    }
    
    
    fig.add_trace(go.Candlestick(**candlestick), row=1, col=1)
    fig.add_trace(go.Bar(**volume), row=2, col=1)
    plot_color = "rgba(150, 150, 150, .01)"

    fig.update_layout(
        bargap= 0.2,
        margin= dict(l=0, r=0, t=0, b=0),
        xaxis_rangeslider_visible = False,      
        xaxis_rangebreaks = [{'bounds': ['sat', 'mon']}],
        xaxis_showgrid = False,
        yaxis_showgrid = False,
        paper_bgcolor = plot_color,
        plot_bgcolor= plot_color,
    )
    
    config = {
        'modeBarButtonsToAdd': ['drawline']
    }
    
    st.plotly_chart(fig, use_container_width=use_container_width, config=config)    

def render_dataframe(df, key, selection_mode=None):
    builder = GridOptionsBuilder.from_dataframe(df)
    
    builder.configure_default_column(
        min_column_width = 5,
        editable=False, 
        resizable=True, 
        sorteable=True, 
        groupable=False,
        value=True, 
        enableRowGroup=True, 
        aggFunc="count"
    )
    
    if selection_mode:
        builder.configure_selection(
            selection_mode=selection_mode, 
            use_checkbox=True, 
            pre_selected_rows=[], 
            rowMultiSelectWithClick=False, 
            suppressRowDeselection=False, 
            groupSelectsChildren=True, 
            groupSelectsFiltered=True
        )
    
    gridOptions = builder.build()
    
    aggrid = AgGrid(
        df, 
        key=key, 
        gridOptions=gridOptions, 
        height = 400, 
        width = "100%", 
        enable_enterprise_modules = True,
        reload_data=True,
        update_mode= GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.MODEL_CHANGED | GridUpdateMode.VALUE_CHANGED,
        conversion_errors = "coerce"
    )
    
    return aggrid
