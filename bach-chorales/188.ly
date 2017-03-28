\version "2.10.33"

\header {
  title = "188 - Ich dank' dir schon durch deinen Sohn"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key f \major
}


soprano = \relative c' {
  f4 f4 f4 
  bes2 g4 
  a2 b4 
  c2. 
  a4 bes4 c4   %5
  d2 bes4 
  c2. 
  bes2. 
  c4 a8 bes8 c4 
  d2 c4   %10
  c4. bes8 a4 
  g2. 
  c4 c4 c4 
  bes2 a4 
  a4 g2   %15
  f2. 
  
}

alto = \relative c' {
  c4 c4 f4~ 
  f4 e8 d8 e4 
  f2 f4 
  g2. 
  f4 f4 g8 a8   %5
  bes2 bes4~ 
  bes2 a4 
  f2. 
  g4 f4 f4 
  f2 f4   %10
  g8 f8 e4 f4~ 
  f4 e2 
  f4 g8 bes8 a4 
  g4 f8 e8 f4 
  f2 e4   %15
  c2. 
  
}

tenor = \relative c' {
  a4 a4 a4 
  g8 a8 bes4 c4 
  c2 d4 
  e2. 
  c4 d4 ees4   %5
  f2 f4 
  g8 f8 ees8 d8 c8 ees8 
  ees8 d16 c16 d2 
  c4 c4 c4~ 
  c4 bes4 a4   %10
  g4 c4 c4 
  c2. 
  c4 c4 f8 e8 
  d4 c4 c4 
  c4 d4 c8 bes8   %15
  a2. 
  
}

baixo = \relative c {
  f4 f8 e8 d4 
  g4 g,4 c4 
  f4 e4 d4 
  c2. 
  f8 ees8 d4 c4   %5
  bes4 c4 d4 
  ees4 c4 f4 
  bes,2. 
  e4 f4 a,4 
  bes8 c8 d8 e8 f4   %10
  e4 c4 f4 
  c2. 
  a'4 e4 f4 
  g4 c,4 f4 
  a,4 bes4 c4   %15
  f,2. 
  
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