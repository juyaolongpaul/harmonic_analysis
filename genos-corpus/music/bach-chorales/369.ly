\version "2.10.33"

\header {
  title = "369 - Jesu, der du meine Seele"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key f \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    f4 f4 c4 des4 
    ees4 des8 c8 c4 bes4 
    des4 des4 c4 bes4 
    a4 bes4 c2 }
  c4 des4 ees8 des8 c4   %5
  des4 c8 bes8 bes4 aes4 
  des4 ees4 f8 ees8 des4 
  ges4 f4 ees4 des4 
  c4 des4 ees4 f4 
  ees4 des4 c2   %10
  bes4 c4 des4 ees8 des8 
  c4. c8 bes2 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    bes4 f8 g8 a4 f4 
    ees8 f8 ges4 f8 ees8 des4 
    f4 bes8 aes8 ges8 f8 f8 ees8 
    ees8 c8 f8 g8 a2 }
  a4 bes4 bes4 aes4   %5
  aes8 g8 aes4 g4 ees4 
  bes'4 aes4 aes4 bes8 aes8 
  bes8 c8 des4 aes8 ges8 f4 
  aes4 aes4 aes4 aes8 bes8 
  c4 bes4 a2   %10
  bes4 f4 f4 ees8 f8 
  ges4 f8 ees8 d2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    des8 c8 bes4 f'8 ees8 des8 c8 
    bes4 c4 a4 f4 
    bes4 bes4 bes8 a8 bes4 
    c4 bes8 des8 c2 }
  f4 f4 ees4 ees4   %5
  des8 ees8 f4 ees8 des8 c4 
  f8 ges8 f8 ees8 des8 c8 bes4 
  ees4 aes,8 bes8 c4 des4 
  ees4 f4 ees4 des4 
  ges8 f8 f8 e8 f2   %10
  f4 f4 bes,4 bes4 
  bes4 a4 f2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    bes8 c8 des8 ees8 f4 bes8 aes8 
    ges4 f8 ees8 f4 bes,4 
    bes'8 aes8 ges8 f8 ees8 f8 ges4 
    c,8 ees8 des8 bes8 f'2 }
  f4 bes8 aes8 g8 ees8 aes8 g8   %5
  f8 ees8 des4 ees4 aes,4 
  bes4 c4 des4 ges8 f8 
  ees4 f8 ges8 aes4 des,4 
  aes'8 ges8 f8 ees8 des8 c8 des4 
  a4 bes4 f'2   %10
  des'8 c8 bes8 a8 bes8 aes8 ges8 f8 
  ees4 f4 bes,2 
  
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