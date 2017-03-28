\version "2.10.33"

\header {
  title = "192 - Gottlob, es geht nunmehr zu Ende"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key bes \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    d4 bes4 d4 
    ees2 d4 
    c2 bes4 
    c2 c4 
    d4 d4 d4     %5
    g,2 a4 
    bes2 bes4 
    a2. }
  f4 c'4 c4 
  c2 d4   %10
  ees2 ees4 
  d2. 
  f4 f4 f4 
  ees2. 
  d4 d4 c4   %15
  bes2. 
  
}

alto = \relative c' {
  \repeat volta 2 {
    f4 f4 bes4~ 
    bes4 a4 f4 
    g4 f4. e8 
    a2 a4 
    bes4 f8 a8 g8 f8     %5
    e2 f4~ 
    f4 e8 d8 e4 
    f2. }
  c4 c4 e4 
  f2 bes4~   %10
  bes4 a8 g8 a4 
  bes2. 
  f4 bes8 a8 bes4 
  g4 c4 a4 
  bes4 bes4. a8   %15
  f2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    bes4 bes4 f'4 
    ees4 c4 bes4~ 
    bes4 a4 bes4 
    f'2 f4 
    f4 bes,8 c8 d4     %5
    c2 c4 
    d4 bes4 c4 
    c2. }
  a4 g4 g4 
  a2 f'4   %10
  g4 ees4 f4 
  f2. 
  d4 bes8 c8 d4 
  c2 f4 
  f2 f8. ees16   %15
  d2. 
  
}

baixo = \relative c {
  \repeat volta 2 {
    bes4 d4 bes4 
    c4 f4 bes4 
    e,4 f4 g4 
    f2 f4 
    bes,4 d4 bes4     %5
    c4 bes4 a4 
    g2 c4 
    f,2. }
  f'4 e4 c4 
  f4 ees4 d4   %10
  c2 f4 
  bes,2. 
  bes4 d4 bes4 
  c4 a4 f4 
  bes4 d4 f4   %15
  bes,2. 
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}